import os
import yaml
import torch
import argparse
from pathlib import Path
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import wandb
import numpy as np

# Model imports
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.loss import ComputeLoss
from utils.metrics import ap_per_class
from utils.general import (
    check_dataset, colorstr, increment_path,
    one_cycle, set_logging, init_seeds
)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt')
    parser.add_argument('--data', type=str, default='data/coco128.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--device', default='0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--adam', action='store_true', help='use Adam optimizer')
    return parser.parse_args()

def train():
    opt = parse_opt()
    set_logging()
    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    
    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=getattr(opt, 'exist_ok', False))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B
    wandb.init(project=opt.project, name=opt.name, config=vars(opt))
    
    # Load dataset
    data_dict = check_dataset(opt.data)
    nc = int(data_dict['nc'])
    names = data_dict['names']
    
    # Model
    model = Model(cfg=None, ch=3, nc=nc).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001) if opt.adam else optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    
    # Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs) if opt.cos_lr else None
    
    # Dataloaders
    train_loader = create_dataloader(data_dict['train'], opt.img_size, opt.batch_size, 
                                   augment=True, workers=opt.workers)
    
    # Training loop
    for epoch in range(opt.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{opt.epochs-1}')
        
        for i, (imgs, targets, paths, _) in enumerate(pbar):
            imgs = imgs.to(device) / 255.0
            
            # Forward
            pred = model(imgs)
            loss = model.compute_loss(pred, targets.to(device))[0]
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            pbar.set_postfix(loss=loss.item())
            wandb.log({'train/loss': loss.item()})
        
        if scheduler:
            scheduler.step()
        
        # Save model
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir / f'model_epoch{epoch}.pt')
    
    wandb.finish()

if __name__ == '__main__':
    train()
