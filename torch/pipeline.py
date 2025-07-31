import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import transforms
import torchvision.datasets as datasets
import torchvision.models as models
import deepspeed
from deepspeed.pipe import PipelineModule
# for compression
import pipelineengine
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--model', default='vgg16', type=str, help='model architecture')
parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
parser.add_argument('-s', '--steps', type=int, default=1000, help='quit after this many steps')
parser.add_argument('-p', '--pipeline-parallel-size', type=int, default=2, help='pipeline parallelism')
parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')
parser.add_argument('--seed', type=int, default=42, help='PRNG seed')
parser.add_argument("--freq", default=10, type=int, help='how many iteration to save a full checkpoint')
parser.add_argument("--compressor", default="topk", type=str, help='which compressor to use')
parser.add_argument("--compressor_ratio", default=0.01, type=float, help='choose compress ratio for compressor')
parser.add_argument("--save-dir", default='/data/lowdiff', type=str, help='directory to save checkpoints')


parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers

def main():
    deepspeed.init_distributed()
    dist.barrier()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed.")

    if args.dataset=="imagenet":
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
        
        if args.model == "vgg16":
            model = models.vgg16_bn().cuda()
        elif args.model == "alexnet":
            model = models.AlexNet().cuda()
    
    elif args.dataset=="cifar10":
        dataset = datasets.CIFAR10(
                '/data/dataset/cv/cifar10',
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
        )
        if args.model == "vgg16":
            model = models.vgg16_bn(num_classes=10).cuda()
        elif args.model == "alexnet":
            model = models.AlexNet(num_classes=10).cuda()
    
    model = PipelineModule(layers=join_layers(model),
                         loss_fn=torch.nn.CrossEntropyLoss(),
                         num_stages=args.pipeline_parallel_size,
                         activation_checkpoint_interval=0)

    ds_config = {
    "train_batch_size" : 64,
    "train_micro_batch_size_per_gpu" : 2,

    "optimizer": {
        "type": "Adam",
        "params": {
        "lr": 0.001,
        "betas": [
            0.9,
            0.999
        ],
        "eps": 1e-8
        }
    },
    
    "steps_per_print" : 10
    }

    model, _, _, _ = pipelineengine.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config)

    # model.pipeline_enable_backward_allreduce = False
    
    for step in range(args.steps):
        loss = model.train_batch()
        
        if dist.get_rank() == 0 and args.freq > 0 and step % args.freq == 0:
                        begin_full = time.time()
                        torch.save({
                            'model': model.module.state_dict(),
                            'optimizer' : model.optimizer.state_dict(),
                        }, '{}/{}_{}_{}_{}_pipe.pth.tar'.format(args.save_dir,args.model,args.dataset,args.compressor,args.compressor_ratio,step))
                        end_full = time.time()
                        print("base checkpoint takes {:.3f}s".format(end_full - begin_full))
        
    
if __name__ == '__main__':
    main()
    
