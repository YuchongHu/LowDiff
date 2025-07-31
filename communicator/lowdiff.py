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
    def __init__(self, model, k=0.01, num_threads=None, save_batch_freq=1):
        """
        Initialize the Communicator for Top-K gradient compression with async all_gather.

        Args:
            model (nn.Module): The PyTorch model.
            k (float): Compression ratio (top-k percentage of gradient to keep).
            num_threads (int, optional): Number of threads for decompression. 
                                          Defaults to half of CPU cores.
            batch (int): In-memory batching frequency for saving compressed gradients.
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
        
        if dist.get_rank() == 0:
            self.save_batch_freq = save_batch_freq
            self.diff_ckpt = {}
            self.queue = mp.Queue()
            self.save_process = mp.Process(target=diff_ckpt_saver, args=(self.queue,self.save_batch_freq))
            self.save_process.start()
            print("save process start!")
        
        
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
        
        if dist.get_rank() == 0:
            self.diff_ckpt[param_name] = {'values': gathered_values, 'indices': gathered_indices, 'shape': grad.shape}
        
        return None  # Do not modify grad immediately
    
    def register_hooks(self):
        """
        Register Top-K compression hooks for model parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: self.async_send(grad, name))
    
    def decompress_save(self, diff, filename, i):
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

        # Clear stored data
        self.compression_data.clear()
        
        # Send the compressed gradients to the save process
        if diff and dist.get_rank() == 0:
            self.queue.put((self.diff_ckpt,filename,i))

    def __del__(self):
        """
        Ensure the thread pool is properly shut down on object destruction.
        """
        self.executor.shutdown(wait=True)
        self.queue.put(None)
        self.save_process.join()

def diff_ckpt_saver(queue,save_batch_freq):
    """
    Background process that saves compressed gradients to disk.
    
    Args:
        queue (mp.Queue): Queue receiving data to be saved.
        save_batch_freq (int): Save frequency in terms of batch steps.
    """
    
    batch_buffer = {}
    print("batching freq = {}".format(save_batch_freq))
    
    while True:
        data = queue.get()
        
        if data is None:
            break
        diff, filename, i = data
        data = _to_cpu(data)
    
        if save_batch_freq == 1 :
            begin = time.time()
            torch.save(diff, filename)
            end = time.time()
            now = datetime.datetime.now()
            print("Saved {} time: {:.3f}s at {}".format(filename, end - begin, now))
        
        else: 
            batch_buffer[i] = diff
            if i % save_batch_freq == save_batch_freq-1:
                begin = time.time()
                torch.save(batch_buffer, filename)
                end = time.time()
                print("Saved {} time: {:.3f}s".format(filename, end - begin))
                batch_buffer={}

def _to_cpu(data):
    """
    Move tensor to CPU and return
    """
    if hasattr(data, 'cpu'):
        cpu_data = data.cpu().clone()
        return cpu_data
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    else:
        return data
