import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import deepspeed
from deepspeed import comm as dist
from communicator.lowdiff import Communicator
import torch.multiprocessing as mp
mp.set_start_method('spawn',force=True)
import re

# Argument parsing
parser = argparse.ArgumentParser(description='DeepSpeed ImageNet Training with TopK Compression')
parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name')
parser.add_argument('--model', default='resnet101', type=str, help='model architecture')
parser.add_argument('--epochs', default=1, type=int, help='number of epochs to run')
parser.add_argument('--batch-size', default=64, type=int, help='batch size per GPU')
parser.add_argument('--lr', '--learning-rate', default=0.0125, type=float, dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--workers', default=1, type=int, help='data loading workers')
parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
parser.add_argument('--compress_ratio', default=0.01, type=float, help='TopK compression ratio')
parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
parser.add_argument("--compressor", default="topk", type=str, help='which compressor to use')
parser.add_argument("--compressor_ratio", default=0.01, type=float, help='choose compress ratio for compressor')
parser.add_argument("--save-dir", default='/data/lowdiff', type=str, help='directory to save checkpoints')
parser.add_argument("--resume", type=int, default=0, help='resume from checkpoint')
parser.add_argument("--diff", action="store_true", help='whether to use differentail ckpt')
parser.add_argument("--freq", default=0, type=int, help='how many iteration to save a full checkpoint')
parser.add_argument("--save-batch-freq", default='1', type=int, help='in-memory batching frequency')

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
    optimizer = optim.Adam(model.parameters(), args.lr)
    
    # optionally resume from a checkpoint at rank 0, then broadcast weights to other workers
    if args.resume and dist.get_rank() == 0:
        start_resume = time.time()
        model, optimizer = load_base_checkpoint(model,optimizer)
        if args.save_batch_freq>1:
            model, optimizer = load_batch_differential_checkpoint(model,optimizer)
        else:
            model, optimizer = load_differential_checkpoint(model,optimizer)
        end_resume = time.time()
        print("resume takes {:.3f}s".format(end_resume - start_resume))
        # sync_model_optimizer_state(model, optimizer)
    
    # Define the configuration dictionary directly in code
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1
    }
    
    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=model.parameters(), config=ds_config)
    model.enable_backward_allreduce = False
    
     # Use the Communicator class
    communicator = Communicator(model, k=args.compress_ratio, save_batch_freq=args.save_batch_freq)
    communicator.register_hooks()
    
    criterion = nn.CrossEntropyLoss()
    
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
            communicator.decompress_save(args.diff, '{}/{}_{}_{}_{}_{}-{}_batch{}.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,epoch,batch_idx,args.save_batch_freq), batch_idx)
            model.step()

            if dist.get_rank() == 0:
                print("[Epoch {}/{}] Batch {}, Loss: {:.3f}, Time: {:.3f}"
                    .format(epoch, args.epochs, batch_idx, loss.item(), time.time() - end))

            if dist.get_rank() == 0 and args.freq > 0 and batch_idx % args.freq == 0:
                        begin_full = time.time()
                        torch.save({
                            'epoch': epoch + 1,
                            'model': model.module.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, '{}/{}_{}_{}_{}_{}_{}_full.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,epoch,batch_idx))
                        end_full = time.time()
                        print("base checkpoint takes {:.3f}s".format(end_full - begin_full))

            end = time.time()

        print(f"Epoch {epoch} completed.")

def load_base_checkpoint(model, optimizer):
    start = time.time()
    filedir = args.save_dir
    filepath = filedir + '/' + args.model + '_' + args.dataset + '_' + args.compressor + '_' + str(args.compressor_ratio) + '_' + str(args.resume-1) + '_0_full' + '.pth.tar'
    if os.path.isfile(filepath):
        print("loading {}".format(filepath))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        end = time.time()
        print("load base checkpoint takes {:.3f}s".format(end - start))
        return model, optimizer
    else:
        raise ValueError("No checkpoint found")

def topk_decompress(values, indices, shape):
    tensor_decompressed = torch.zeros(shape).cuda().view(-1)
    for idx, val in zip(indices, values):
        tensor_decompressed = tensor_decompressed.scatter_add_(0, idx, val)
    return tensor_decompressed.view(shape)

def find_max():
    files = os.listdir(args.save_dir)
    if args.save_batch_freq>1:
        pattern = r'{}_{}_{}_{}_{}-(\d+)_batch{}\.pth\.tar'.format(args.model, args.dataset, args.compressor, args.compressor_ratio, args.resume-1, args.save_batch_freq)
    else:
        pattern = r'{}_{}_{}_{}_{}-(\d+)_batch1\.pth\.tar'.format(args.model, args.dataset, args.compressor, args.compressor_ratio, args.resume-1)
    max_x = -1
    for file in files:
        match = re.match(pattern, file)
        if match:
            x = int(match.group(1))
            if x > max_x:
                max_x = x
    if max_x != -1:
        print("Max diff ckpt at epoch {}, iteration {}".format(args.resume, max_x))
    else:
        print("no diff ckpt found")
    return max_x
    
def load_differential_checkpoint(model, optimizer):
    filedir = args.save_dir
    _parameter_names = {name: param for name, param in model.named_parameters()}
    # skip synchronize
    # with optimizer.skip_synchronize():
    iterations = find_max()
    for i in range(0, iterations):
        filepath = filedir + '/{}_{}_{}_{}_{}-{}_batch1.pth.tar'.format(args.model, args.dataset, args.compressor, args.compressor_ratio, args.resume-1, i)
        tensor_compressed = torch.load(filepath)
        for key in tensor_compressed.keys():  # the name for trainable params
            tensor = topk_decompress(tensor_compressed[key]['values'], tensor_compressed[key]['indices'], tensor_compressed[key]['shape'])
            param = _parameter_names.get(key)
            if param is not None:
                param.grad = tensor
        optimizer.step()
    return model, optimizer

def load_batch_differential_checkpoint(model, optimizer):
    filedir = args.save_dir
    _parameter_names = {name: param for name, param in model.named_parameters()}
    iterations = find_max()
    for i in range(args.save_batch_freq-1, iterations, args.save_batch_freq):
        filepath = filedir + '/{}_{}_{}_{}_{}-{}_batch{}.pth.tar'.format(args.model, args.dataset, args.compressor, args.compressor_ratio, args.resume-1, i, args.save_batch_freq)
        tensor_compressed = torch.load(filepath)
        for j in range(0, args.save_batch_freq):
            for key in tensor_compressed[i-args.save_batch_freq+j+1].keys():  # the name for trainable params
                tensor = topk_decompress(tensor_compressed[i-args.save_batch_freq+j+1][key]['values'], tensor_compressed[i-args.save_batch_freq+j+1][key]['indices'], tensor_compressed[i-args.save_batch_freq+j+1][key]['shape'])              
                param = _parameter_names.get(key)
                if param is not None:
                    param.grad = tensor
            optimizer.step()
            print("loaded interation {}".format(i))
    return model, optimizer


# bugs need be fixed
def sync_model_optimizer_state(model, optimizer):
    # Only execute synchronization on rank 0
    if dist.get_rank() == 0:
        # Broadcast model parameters
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        # Broadcast optimizer state
        for group in optimizer.param_groups:
            for param in group['params']:
                dist.broadcast(param.data, src=0)
                
    # Synchronize across all ranks
    dist.barrier()
    
if __name__ == '__main__':
    main()
