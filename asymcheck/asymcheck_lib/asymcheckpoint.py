import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import queue
import os


class AsymCheckpoint:
    def __init__(self,
                 ckpt_size =None,
                 warmup = 10,
                 parallel_threads = 32,
                 ratio = 0.01,
                 opt_batch_size = 64 * 1024*1024,
                 ):

        self.ckpt_size = ckpt_size
        self.warmup = warmup
        self.parallel_threads = parallel_threads
        self.ratio = ratio
        self.opt_batch_size = opt_batch_size
        self.iteration = 9

        self.flush_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=32)
        self.shutdown_event = multiprocessing.Event()
        self.flushing_process: Optional[multiprocessing.Process] = None

        self.profiling_data: Dict[str, List[float]] = {
            "fw_start_times": [], "fw_comm_end_times": [],
            "bw_start_times": [], "bw_comm_end_times": [],
        }
        self.checkpointing_data: Dict[str, Any] = {
            "partition": None,
            "cp_params": None,
            "pre_ckpt": {},  
            "batch": {
                "cp_chunks": [],
                "cur_batch_size": 0
            }
        }
        self.partitioning_complete = False
        self.current_checkpoint_offset = 0 
    

    def start_flushing_worker(self):
        if self.flushing_process is None or not self.flushing_process.is_alive():
            self.flushing_process = multiprocessing.Process(
                target=self._flushing_worker_process,
                args=(self.flush_queue, self.shutdown_event)
            )
            
            self.flushing_process.start()

    def stop_flushing_worker(self):
        if self.flushing_process and self.flushing_process.is_alive():
            self.shutdown_event.set()
            self.flushing_process.join()

    @staticmethod
    def _flushing_worker_process(queue, shutdown_event):
        checkpoint_counter = 0
        while not shutdown_event.is_set():
            try:
                data_to_flush = queue.get(timeout=1)
                checkpoint_counter += 1
                        
                with open(f"./checkpoint_batch_{checkpoint_counter}.bin", 'wb') as f:
                    torch.save(data_to_flush, f)
            except queue.Empty:
                continue
            
    def profile_idle_periods(self, iteration):
        if iteration < self.warmup:
            return False 

        self.forward_idle_times = []

        for i in range(len(self.profiling_data["fw_start_times"])):
            idle_start = self.profiling_data["fw_comm_end_times"][i]
            idle_end = self.profiling_data["fw_start_times"][i]
            idle_time = idle_end - idle_start
            self.forward_idle_times.append(idle_time)
    
        self.backward_idle_times = []
        
        bd_start_times = self.profiling_data.get("bw_start_times", [])
        for i in range(len(bd_start_times)):
            if i < len(self.profiling_data["bw_comm_end_times"]):
                idle_start = self.profiling_data["bw_comm_end_times"][i]
                idle_end = bd_start_times[i]
                idle_time = idle_end - idle_start
                self.backward_idle_times.append(idle_time)
    
        return self.forward_idle_times, self.backward_idle_times



    def _profile_overheads(self):
        params = {}
        test_sizes = [64 * 1024, 256 * 1024, 1024 * 1024, 256* 1024 * 1024]
        compression_times, write_times = [], []

        for size in test_sizes:
            num_elements = size//4
            dummy_data = torch.randn(num_elements, device='cpu', dtype=torch.float32)
            start = time.time()
            k = int(num_elements * self.ratio)
            torch.topk(torch.abs(dummy_data), k)
            compression_times.append(time.time() - start)

            start = time.time()
            dummy_data.numpy().tobytes()
            write_times.append(time.time() - start)

        x = np.array(test_sizes, dtype=np.float64)
        A = np.vstack([x, np.ones(len(x))]).T
        beta_c, alpha_c = np.linalg.lstsq(A, np.array(compression_times), rcond=None)[0]
        beta_w, alpha_w = np.linalg.lstsq(A, np.array(write_times), rcond=None)[0]

        params.update({'alpha_c': alpha_c, 'beta_c': beta_c, 'alpha_w': alpha_w, 'beta_w': beta_w})
        
        return params


    def _calculate_partition_sizes(self,iteration):
        self.profile_idle_periods(iteration)

        T_f = sum(self.forward_idle_times)
        
        T_b = sum(self.backward_idle_times)

        total_idle_time = T_f + T_b
        
        if total_idle_time <= 1e-9: 
            return False

        R = self.ckpt_size
        forward_partition_sizes = [int(R * (t / total_idle_time)) for t in self.forward_idle_times]
        backward_partition_sizes = [int(R * (t / total_idle_time)) for t in self.backward_idle_times]
        
        forward_partition_sizes[-1] = R - sum(forward_partition_sizes[:-1])
        backward_partition_sizes[-1] = R - sum(backward_partition_sizes[:-1])

        self.checkpointing_data["partition"] = {
            "forward": forward_partition_sizes,
            "backward": backward_partition_sizes
        }

        self.checkpointing_data["cp_params"] = self._profile_overheads()
        self.partitioning_complete = True

        return True

    def _should_compress(self, partition_size):
        params = self.checkpointing_data["cp_params"]
        if not params:
            return False, 0, 0

        alpha_c, beta_c = params['alpha_c'], params['beta_c']
        alpha_w, beta_w = params['alpha_w'], params['beta_w']
        t_write_uncompressed = alpha_w + beta_w * partition_size
        t_compress = alpha_c + beta_c * partition_size
        compressed_size = int(partition_size * (self.ratio * 2))
        t_write_compressed = alpha_w + beta_w * compressed_size
        return (t_compress + t_write_compressed < t_write_uncompressed,
                t_compress + t_write_compressed, t_write_uncompressed)

    @staticmethod
    def _compress_chunk(chunk, ratio):
        chunk_clone = chunk.clone().cpu()
        chunk_flat = chunk_clone.flatten()
        k = max(1, int(chunk_flat.numel() * ratio))
        values, indices = torch.topk(torch.abs(chunk_flat), k)
        return {'values': values, 'indices': indices, 'original_shape': chunk.shape}

    def _parallel_compress_data(self, data):
        num_chunks = self.parallel_threads
        chunk_size = data.numel() // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else data.numel()

            chunk_view = torch.narrow(data.flatten(), 0, start, end - start)
            chunks.append(chunk_view)

        with ThreadPoolExecutor(max_workers=num_chunks) as executor:
            return list(executor.map(
                lambda c: self._compress_chunk(c, self.ratio), chunks
            ))

    def _enqueue_for_flushing(self, data, is_compressed_batch):
        if is_compressed_batch:
            chunk_size_bytes = data['values'].numel() * data['values'].element_size() + \
                               data['indices'].numel() * data['indices'].element_size()

            if (self.checkpointing_data["batch"]["cur_batch_size"] + chunk_size_bytes) < self.opt_batch_size:
                self.checkpointing_data["batch"]["cp_chunks"].append(data)
                self.checkpointing_data["batch"]["cur_batch_size"] += chunk_size_bytes

                return
            else:
                if self.checkpointing_data["batch"]["cp_chunks"]:
                    batch_to_send = {
                        'type': 'compressed_batch',
                        'data': self.checkpointing_data["batch"]["cp_chunks"]
                    }
                    self.flush_queue.put(batch_to_send)

                    self.checkpointing_data["batch"]["cp_chunks"] = []
                    self.checkpointing_data["batch"]["cur_batch_size"] = 0

            batch_to_send = {'type': 'compressed_batch', 'data': [data]}
            self.flush_queue.put(batch_to_send)
            self.checkpointing_data["batch"]["cur_batch_size"] = chunk_size_bytes

        else:

            data_to_send = {'type': 'uncompressed', 'data': data.clone().cpu()}
            self.flush_queue.put(data_to_send)


    def async_checkpoint(self, iteration, partition_type, partition_idx, data):

        if iteration == self.warmup - 1:
            self._calculate_partition_sizes(self.iteration)

        if not self.partitioning_complete:
            return

        sizes = self.checkpointing_data["partition"][partition_type]
        if partition_idx >= len(sizes):
            return
        
        partition_size = sizes[partition_idx]
        
        total_elements_needed = self.current_checkpoint_offset + partition_size
        if total_elements_needed > data.numel():

            self.current_checkpoint_offset = 0
            return

        data_flat = data.flatten()
        partition_view = torch.narrow(data_flat, 0, self.current_checkpoint_offset, partition_size)
        
        self.current_checkpoint_offset += partition_size
        
        if self.current_checkpoint_offset >= self.ckpt_size:
            self.current_checkpoint_offset = 0

        prev_key = f"{partition_type}_{partition_idx}"
        prev_data = self.checkpointing_data["pre_ckpt"].get(prev_key)
        
        if prev_data is None or prev_data.numel() != partition_view.numel():
            delta_data = partition_view
        else:
            delta_data = partition_view - prev_data.to(partition_view.device)
            
        self.checkpointing_data["pre_ckpt"][prev_key] = partition_view.clone().cpu()

        compress, _, _ = self._should_compress(partition_size)

        if compress:
            compressed_chunks = self._parallel_compress_data(delta_data, self.parallel_threads, ratio)
            for chunk in compressed_chunks:
                self._enqueue_for_flushing(chunk, is_compressed_batch=True)
        else:
            self._enqueue_for_flushing(delta_data, is_compressed_batch=False)
    
    
    @staticmethod
    def save_full_checkpoint(model, optimizer, epoch, path = "./full_checkpoint.pt"):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
   


