import time
import copy
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
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Define the configuration dictionary directly in code
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1
    }
    
    # Initialize DeepSpeed
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, model_parameters=model.parameters(), config=ds_config)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, targets) in enumerate(train_loader):
            begin = time.time()
            images, targets = images.cuda(), targets.cuda()
            output = model(images)
            loss = criterion(output, targets)
            model.backward(loss)
            model.step()
            end = time.time()
            if dist.get_rank() == 0:
                print("iteration takes {:.3f}s".format(end - begin))

            # save full ckpt
            if dist.get_rank() == 0 and args.freq > 0 and batch_idx % args.freq == 0:
                        begin_full = time.time()
                        torch.save({
                            'epoch': epoch + 1,
                            'model': model.module.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, '{}/{}_full_optimizer.pth.tar'.format(args.save_dir,args.model))
                        end_full = time.time()
                        print("base checkpoint takes {:.3f}s".format(end_full - begin_full))
            
            # save gradient for observe gradient size
            if dist.get_rank() == 0 and batch_idx ==0 :
                begin = time.time()
                gradients = {}
                # Save gradients for each parameter in the model
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.cpu()
                torch.save(gradients, '{}/{}_gradients.pth.tar'.format(args.save_dir,args.model))
                end = time.time()
                print("save gradients takes {:.3f}s".format(end - begin))
                 
            if batch_idx !=0 and dist.get_rank() == 0: 
                begin = time.time()
                # compress_diff = compress_model_and_optimizer(model.module,prev_model,optimizer.state_dict(),prev_optimizer)
                compress_diff = compress_model(model.module,prev_model)
                
            
            if dist.get_rank() == 0:
                # prev_model = copy.deepcopy(model.module).cpu()
                prev_model = copy.deepcopy(model.module)
                # prev_optimizer = _to_cpu(copy.deepcopy(model.optimizer.state_dict()))
                # prev_optimizer = copy.deepcopy(model.optimizer.state_dict())
            
            if batch_idx !=0 and dist.get_rank() == 0: 
                end = time.time()
                print("compress model takes {:.3f}s".format(end - begin))
                
            if batch_idx !=0 and dist.get_rank() == 0: 
                begin = time.time()
                # torch.save(compress_diff, '{}/{}_diff.pth.tar'.format(args.save_dir,args.model))
                torch.save((compress_diff,model.optimizer.state_dict()), '{}/{}_diff.pth.tar'.format(args.save_dir,args.model))
                end = time.time()
                print("save diff takes {:.3f}s".format(end - begin))

        print(f"Epoch {epoch} completed.")
    
def topk_compress(tensor):
    """
    Compress the gradient into Top-K format.
    """
    num_elements = tensor.numel()
    shape = tensor.size()
    k_elements = max(1, int(num_elements * 0.01))

    values, indices = torch.topk(tensor.view(-1).abs(), k_elements, sorted=False)
    values = tensor.view(-1).gather(0, indices)

    return indices, values, shape

def topk_decompress(values, indices, shape):
    tensor_decompressed = torch.zeros(shape).cuda().view(-1)
    for idx, val in zip(indices, values):
        tensor_decompressed = tensor_decompressed.scatter_add_(0, idx, val)
    return tensor_decompressed.view(shape)


def _to_cpu(data):
    """
    Move tensor to CPU and return
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

def _to_cuda(data):
    """
    Move tensor to cuda and return
    """
    if hasattr(data, 'cuda'):
        return data.detach().cuda().clone()
    elif isinstance(data, dict):
        return {k: _to_cuda(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_to_cuda(v) for v in data]
    elif isinstance(data, tuple):
        return tuple(_to_cuda(v) for v in data)
    else:
        return data

def compress_model(model_a, model_b):
    model_b = model_b.cuda()
    compressed_data = {
        'model': {},
    }
    for (name, param_a) in model_a.state_dict().items():
        param_b = model_b.state_dict()[name]
        if torch.is_tensor(param_a):
            diff = (param_a - param_b).to(param_a.device)
            indices, values, shape = topk_compress(diff)
            compressed_data['model'][name] = {
                'indices': indices,
                'values': values,
                'shape': shape
            }
    
    return compressed_data

def compress_model_and_optimizer(model_a, model_b, optimizer_a, optimizer_b):
    model_b = model_b.cuda()
    optimizer_b = _to_cuda(optimizer_b)
    
    compressed_data = {
        'model': {},
        'optimizer': {}
    }

    for (name, param_a) in model_a.state_dict().items():
        param_b = model_b.state_dict()[name]
        if torch.is_tensor(param_a):
            diff = (param_a - param_b).to(param_a.device)
            indices, values, shape = topk_compress(diff)
            compressed_data['model'][name] = {
                'indices': indices,
                'values': values,
                'shape': shape
            }

    state_a = optimizer_a['state']
    state_b = optimizer_b['state']

    for group_idx in state_a.keys():
        compressed_data['optimizer'][group_idx] = {}
        for state_key, state_val_a in state_a[group_idx].items():
            state_val_b = state_b[group_idx][state_key]
            if torch.is_tensor(state_val_a):
                diff = (state_val_a - state_val_b).to(state_val_a.device)
                indices, values, shape = topk_compress(diff)
                compressed_data['optimizer'][group_idx][state_key] = {
                    'indices': indices,
                    'values': values,
                    'shape': shape
                }
            else:
                compressed_data['optimizer'][group_idx][state_key] = state_val_a

    return compressed_data
    
if __name__ == '__main__':
    main()
