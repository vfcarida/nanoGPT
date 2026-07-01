"""
Optimized data loader utilizing background prefetch threads.
Designed to hide disk I/O and CPU-to-GPU transfer latency behind model execution.
"""

import os
import time
import queue
import threading
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)

class PrefetchDataLoader:
    """
    PrefetchDataLoader speeds up training and validation batch preparation by:
    1. Caching the numpy memmap descriptor to prevent opening/closing files constantly.
    2. Spawning background prefetcher threads to prepare batches concurrently.
    3. Pinning batch tensors on CPU to enable asynchronous, non-blocking transfer to CUDA device.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        block_size: int,
        device: str,
        prefetch_factor: int = 4
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.device_type = 'cuda' if 'cuda' in device else 'cpu'
        self.prefetch_factor = prefetch_factor
        
        self.memmaps: Dict[str, np.memmap] = {}
        self.queues: Dict[str, queue.Queue] = {
            'train': queue.Queue(maxsize=prefetch_factor),
            'val': queue.Queue(maxsize=prefetch_factor)
        }
        
        self.running = True
        self.lock = threading.Lock()
        self.threads: Dict[str, threading.Thread] = {}
        
        # Initialize splits
        for split in ['train', 'val']:
            # Cache memmap in main thread to detect errors early
            self._get_memmap(split)
            
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(split,),
                name=f"PrefetchWorker-{split}",
                daemon=True
            )
            self.threads[split] = thread
            thread.start()

    def _get_memmap(self, split: str) -> np.memmap:
        with self.lock:
            if split not in self.memmaps:
                file_path = os.path.join(self.data_dir, f"{split}.bin")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Data binary not found at: {file_path}")
                self.memmaps[split] = np.memmap(file_path, dtype=np.uint16, mode='r')
            return self.memmaps[split]

    def _prefetch_worker(self, split: str):
        try:
            data = self._get_memmap(split)
        except Exception as e:
            logger.error(f"Failed to initialize memmap for split '{split}': {e}")
            return
            
        while self.running:
            try:
                # Randomly sample batch indices
                ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
                
                # Construct batch
                x_list = []
                y_list = []
                for i in ix:
                    chunk = data[i : i + self.block_size + 1]
                    x_list.append(torch.from_numpy(chunk[:-1].astype(np.int64)))
                    y_list.append(torch.from_numpy(chunk[1:].astype(np.int64)))
                    
                x = torch.stack(x_list)
                y = torch.stack(y_list)
                
                if self.device_type == 'cuda':
                    x = x.pin_memory()
                    y = y.pin_memory()
                    
                # Push batch to queue with timeout to react to thread shutdown
                while self.running:
                    try:
                        self.queues[split].put((x, y), timeout=0.1)
                        break
                    except queue.Full:
                        continue
            except Exception as e:
                # Fail-safe wait if exception occurs (e.g. OS memory mapping bounds)
                if self.running:
                    logger.error(f"Error in dataloader worker '{split}': {e}")
                    time.sleep(1.0)

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets a batch of data for the given split (train/val).
        Attempts to read from prefetch queue, falls back to synchronous extraction if queue is empty.
        """
        if not self.running:
            raise RuntimeError("DataLoader has been shut down.")
            
        try:
            x, y = self.queues[split].get(timeout=5.0)
        except queue.Empty:
            logger.warning(f"DataLoader queue for '{split}' was empty. Loading synchronously.")
            data = self._get_memmap(split)
            ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
            if self.device_type == 'cuda':
                x, y = x.pin_memory(), y.pin_memory()
                
        # Send to target device (asynchronously if CUDA)
        if self.device_type == 'cuda':
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
            
        return x, y

    def close(self):
        """
        Gracefully stop prefetching threads and clear caches.
        """
        self.running = False
        # Empty queues to release blocked worker threads
        for split in self.queues:
            while not self.queues[split].empty():
                try:
                    self.queues[split].get_nowait()
                except queue.Empty:
                    break
        for split, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=1.0)
        with self.lock:
            self.memmaps.clear()
        logger.info("PrefetchDataLoader resource cleanup completed.")
