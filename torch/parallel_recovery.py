import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import threading
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from deepspeed import comm as dist
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))
from communicator.lowdiff import Communicator
import re
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    set_seed
)

# Argument parsing
parser = argparse.ArgumentParser(description='DeepSpeed NLP Training with TopK Compression')
parser.add_argument('--dataset', default='wikitext-2', type=str, help='dataset name')
parser.add_argument('--model', default='gpt2', type=str, help='model architecture')
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
parser.add_argument("--diff", action="store_true", help='whether to use differential checkpoint')
parser.add_argument("--freq", default=0, type=int, help='how many iterations to save a full checkpoint')
parser.add_argument("--save-batch-freq", default='1', type=int, help='in-memory batching frequency')
parser.add_argument("--seq_length", type=int, default=512)  
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--parellel_recovery", type=bool, default=True, help='whether to use parallel recovery')
args = parser.parse_args()


def main():
    # Initialize argument parsing
    model_path = "/data/dataset/nlp/openai-community/" + args.model

    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()
    dist.barrier()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    set_seed(42 + rank)  # Set deterministic seed
    torch.cuda.set_device(args.local_rank)
    print(f"[Rank {rank}/{world_size}] Initialized DeepSpeed")

    # Load dataset and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print("Tokenizer loaded successfully.")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.seq_length,
            padding="max_length"
        )

    # Load and process the wikitext-103 dataset
    if args.dataset == 'wikitext-103':
        dataset = load_dataset("/data/dataset/nlp/transformer/wikitext-103", 
                        data_files={
                            "train": "/data/dataset/nlp/transformer/wikitext-103/train.txt",
                            "validation": "/data/dataset/nlp/transformer/wikitext-103/valid.txt",
                            "test": "/data/dataset/nlp/transformer/wikitext-103/test.txt"
                        })["train"]
    
    elif args.dataset == 'wikitext-2':
        dataset = load_dataset("/data/dataset/nlp/transformer/wikitext-2", 
                        data_files={
                            "train": "/data/dataset/nlp/transformer/wikitext-2/train.txt",
                            "validation": "/data/dataset/nlp/transformer/wikitext-2/valid.txt",
                            "test": "/data/dataset/nlp/transformer/wikitext-2/test.txt"
                        })["train"]
    else:
        raise ValueError("Incorrect dataset Name")

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=12
    )

    print("Dataset mapping completed successfully.")
    # Data collator (auto-generates labels)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Use causal language modeling
    )

    # Distributed sampler
    train_sampler = DistributedSampler(
        tokenized_dataset,
        shuffle=True,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=4
    )

    # Initialize model (enable gradient checkpointing to save memory)
    print("Loading model...")
    if args.model == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained("/data/dataset/nlp/openai-community/gpt2")
    elif args.model == 'gpt2-medium':
        model = GPT2LMHeadModel.from_pretrained("/data/dataset/nlp/openai-community/gpt2-medium")
    elif args.model == 'gpt2-large':
        model = GPT2LMHeadModel.from_pretrained("/data/dataset/nlp/openai-community/gpt2-large")
    else:
        print("Model loading failed.")
    model.gradient_checkpointing_enable()  
    model.cuda()
    print("Model loaded successfully.")
    
    ds_config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-4,
                "weight_decay": 0.01
            }
        },
    }
    model, optimizer, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)
    # Optionally resume from a checkpoint at rank 0, then broadcast weights to other workers
    if args.resume and dist.get_rank() == 0:

        model, optimizer = load_base_checkpoint(model,optimizer)
        if args.save_batch_freq>1:
            model, optimizer = load_batch_differential_checkpoint(model,optimizer)
        else:
            if args.parellel_recovery:
                model, optimizer = new_load_differential_checkpoint(model,optimizer, args.compress_ratio, args)
            else:
                model, optimizer = load_differential_checkpoint(model,optimizer)
    
    model.cuda()

    # Use the Communicator class
    communicator = Communicator(model, k=args.compress_ratio, save_batch_freq=args.save_batch_freq)
    communicator.register_hooks()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            end = time.time()
            inputs = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

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
                        print("Base checkpoint takes {:.3f}s".format(end_full - begin_full))

            end = time.time()

        print(f"Epoch {epoch} completed.")

def load_base_checkpoint(model, optimizer):
    start = time.time()
    filedir = args.save_dir
    filepath = filedir + '/' + args.model + '_' + args.dataset + '_' + args.compressor + '_' + str(args.compressor_ratio) + '_' + str(args.resume-1) + '_0_full' + '.pth.tar'
    if os.path.isfile(filepath):
