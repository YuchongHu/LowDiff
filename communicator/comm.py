import os
import torch
from deepspeed import comm as dist
# import torch.distributed as dist
import concurrent.futures
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import datetime
import time

class Communicator:
    def __init__(self, model, k=0.01, num_threads=None):
        """
        Initialize the Communicator for Top-K gradient compression with async all_gather.

        Args:
            model (nn.Module): The PyTorch model.
            k (float): Compression ratio (top-k percentage of gradient to keep).
            num_threads (int, optional): Number of threads for decompression. 
                                          Defaults to half of CPU cores.
        """
        self.k = k
        self.model = model
        self.compression_data = {}  # Store async work handles and gathered results
        
        # Get the number of available CPU threads (default to half of total cores, max 32)
        if num_threads is None:
            num_threads = int(os.cpu_count() / 2)
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)  # Thread pool
        self.param_dict = dict(self.model.named_parameters())
        
        print(f"Using {num_threads} threads for gradient decompression.")
        
    def topk_compress(self, tensor):
        """
        Compress the gradient into Top-K format.
        """
        num_elements = tensor.numel()
        k_elements = max(1, int(num_elements * self.k))

        values, indices = torch.topk(tensor.view(-1).abs(), k_elements, sorted=False)
        values = tensor.view(-1).gather(0, indices)

        return indices, values
        
    def async_send(self, grad, param_name):
        """
        Hook function for gradient compression.
        """
        world_size = dist.get_world_size()
        indices, values = self.topk_compress(grad)

        gathered_indices = [torch.zeros_like(indices) for _ in range(world_size)]
        gathered_values = [torch.zeros_like(values) for _ in range(world_size)]
        
        # Perform async all_gather
        work_indices = dist.all_gather(gathered_indices, indices, async_op=True)
        work_values = dist.all_gather(gathered_values, values, async_op=True)
        
        # Store work handles and gathered buffers
        self.compression_data[param_name] = {
            "work_indices": work_indices,
            "work_values": work_values,
            "gathered_indices": gathered_indices,
            "gathered_values": gathered_values,
            "grad_shape": grad.shape
        }
        
        return None  # Do not modify grad immediately
    
    def register_hooks(self):
        """
        Register Top-K compression hooks for model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.async_send(grad, name))
    
    def decompress(self):
        """
        Parallel gradient restoration.
        """
        def process_gradient(param, data):
            data["work_indices"].wait()
            data["work_values"].wait()

            restored_grad = torch.zeros(data["grad_shape"], device=data["gathered_values"][0].device).view(-1)

            for indices, values in zip(data["gathered_indices"], data["gathered_values"]):
                restored_grad.scatter_add_(0, indices, values)  # This remains a CPU/GPU task

            param.grad = restored_grad.view(data["grad_shape"])  # Direct assignment

        # Submit tasks to the thread pool and wait for completion
        futures = [
            self.executor.submit(process_gradient, self.param_dict[name], data)
            for name, data in self.compression_data.items()
        ]
        concurrent.futures.wait(futures)

    def __del__(self):
        """
        Ensure the thread pool is properly shut down on object destruction.
        """
        self.executor.shutdown(wait=True)
        self.save_process.join()