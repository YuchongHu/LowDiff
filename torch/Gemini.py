# We implement Gemini as checkfreq style with Ramdisk for checkpointing

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import deepspeed
from deepspeed import comm as dist
from communicator.comm import Communicator
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
from torch.multiprocessing import Value, Lock

# Argument parsing
parser = argparse.ArgumentParser(description='DeepSpeed ImageNet Training with TopK Compression')
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--model', default='resnet101', type=str, help='Model architecture')
parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='Batch size per GPU')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--workers', default=1, type=int, help='Data loading workers')
parser.add_argument('--seed', type=int, default=42, help='Seed for initializing training')
parser.add_argument("--compressor", default="topk", type=str, help='Which compressor to use')
parser.add_argument("--compressor_ratio", default=0.01, type=float, help='Choose compress ratio for compressor')
parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
parser.add_argument("--save-dir", default='/data/lowdiff', type=str, help='Directory to save checkpoints')
parser.add_argument("--resume", type=int, default=0, help='Resume from checkpoint')
parser.add_argument("--freq", default=0, type=int, help='How many iterations to save a full checkpoint')

args = parser.parse_args()

def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed.")

    # Load dataset
    if args.dataset == 'imagenet':
        dataset = datasets.ImageFolder(
            '/data/dataset/cv/imagenet_0908/train',
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )

    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(
            '/data/dataset/cv/cifar100/train',
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                     std=[0.267, 0.256, 0.276])
            ])
        )

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers)

    # Model selection
    if args.model == 'resnet50':
        model = models.resnet50()
    elif args.model == 'resnet101':
        model = models.resnet101()
    elif args.model == 'vgg16':
        model = models.vgg16_bn()
    elif args.model == 'vgg19':
        model = models.vgg19_bn()
    else:
        print("Model ERROR!")
        return
    
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    # Define the configuration dictionary directly in code
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1
    }
    
    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=model.parameters(), config=ds_config)
    model.enable_backward_allreduce = False
    
    # Use the Communicator class
    communicator = Communicator(model)
    communicator.register_hooks()
    
    criterion = nn.CrossEntropyLoss()

    if dist.get_rank() == 0:
        # Lock initialization
        snapshot = mp.Lock()
        persist = mp.Lock()

        queue = mp.Queue()
        ckpt_process = mp.Process(target=snapshot_persist, args=(queue, snapshot, persist))
        ckpt_process.start()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, targets) in enumerate(train_loader):
            end = time.time()
            images, targets = images.cuda(), targets.cuda()
            output = model(images)
            loss = criterion(output, targets)
            model.backward(loss)
            communicator.decompress()
            
            if dist.get_rank() == 0:
                with snapshot:
                    model.step()
            else:
                model.step()
                
            if dist.get_rank() == 0:
                print("[Epoch {}/{}] Batch {}, Loss: {:.3f}, Time: {:.3f}"
                    .format(epoch, args.epochs, batch_idx, loss.item(), time.time() - end))
            
            if dist.get_rank() == 0:
                got_persist = persist.acquire(block=False)
            if dist.get_rank() == 0 and got_persist:
                queue.put(({
                        'epoch': epoch + 1,
                        'model': model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        }, '{}/{}_{}_{}_{}_{}_{}_full.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,epoch,batch_idx), batch_idx))
                persist.release()
            else:
                print("Pass and wait for persist")
                            
            end = time.time()

        print(f"Epoch {epoch} completed.")

def _to_cpu(data):
    """
    Move tensor to CPU and return.
    """
    if hasattr(data, 'cpu'):
        return data.detach().cpu().clone()
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cpu(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(v) for v in data)
    else:
        return data

def snapshot_persist(queue, snapshot, persist):
    """
    Background process that saves compressed gradients to disk.
    """
    while True:
        data = queue.get()
        
        if data is None:
            return
        data, filename, i = data

        # Snapshot
        with snapshot:
            begin = time.time()
            data = _to_cpu(data)
            end = time.time()
            print("Snapshot {} takes {:.3f}".format(i, end - begin))
        
        # Persist
        with persist:
            begin = time.time()
            torch.save(data, filename)
            end = time.time()
            print("Persist {} takes {:.3f}".format(i, end - begin))
            
    return
    

def save_checkpoint(data, filepath, snapshot, persist):
    """
    Save the checkpoint by spawning a new process.
    """
    chk_process = mp.Process(target=snapshot_persist, args=(data, filepath, snapshot, persist))
    chk_process.start()
    print("Checkpoint process started.")

if __name__ == '__main__':
    main()
