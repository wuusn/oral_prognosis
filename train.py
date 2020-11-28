import numpy as np
import random as rn
import torch
seed = 56
np.random.seed(seed)
rn.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import time
import os
import glob

import argparse

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

# user libraries
import models
from models import *
from modules import *
from validation import *

def train(train_loader,val_loaders,net,device,optimizer,criterion,stopPolicy,logger):
    val = static_val_classification_slide_wise
    n_train = len(train_loader.dataset)

    #before training
    result = val(val_loaders, net, device, criterion, mode='avg')
    print('#before training loss patch-acc slide-auc:', result['loss'], result['patch-accuracy'],result['roc_auc'])

    epoch=0
    try:
        while stopPolicy.check_continue():
            with tqdm(total=n_train, desc=f'epoch: {epoch}') as pbar:
                correct=0
                for batch in train_loader:
                    net.train()
                    x = batch['image'].float().to(device)
                    y = batch['label'].to(device)
                    size = x.size(0)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = net(x)
                        #print(outputs)
                        #print(y)
                        #outputs = torch.sigmoid(outputs)
                        _,preds = torch.max(outputs,1)
                        loss = criterion(outputs, y)
                        loss.backward()
                        optimizer.step()
                        correct += (preds==y).float().sum()
                    stopPolicy.update_step(loss.item()*size)
                    logger.add_step_loss(loss.item())
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(size)

                    #if stopPolicy.check_validation():
                epoch_loss = stopPolicy.update_epoch()
                pbar.set_postfix(**{'loss (epoch)':epoch_loss,'acc (epoch)': correct.item()/n_train})
                result = val(val_loaders, net, device, criterion, mode='avg', showWarn=False)
                print('val loss patch-acc slide-auc:', result['loss'], result['patch-accuracy'], result['roc_auc'])
                logger.add_validation(result, save_model=False)
                stopPolicy.update_validation(result)
                logger.add_epoch_loss(epoch_loss)
            epoch+=1
    except:
        print('break')
    finally:
        logger.save()




if __name__ =='__main__': 
    args = get_args()
    trainLoader = get_patchLoader(args.fold_xls_dir,args.base_dir,args.round,args.fold,'train', args.bs, args.augmentations)
    valTmaLoaders =  get_slide_based_patchLoaders(args.fold_xls_dir,args.base_dir,args.round,args.fold,'val',32,args.augmentations)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    net = getattr(models, f'get_{args.nnmodel}')()
    net = net.float().to(device)
    #optimizer = optim.Adam(net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=.0005, momentum=.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    date = args.date
    codename = args.codename
    logger = Logger(codename+date, args.save_dir)
    stopPolicy = getattr(strategy, f'get_{args.stopby}')(len(trainLoader.dataset),args.epochs)
    train(trainLoader,valTmaLoaders,net,device,optimizer,criterion,stopPolicy,logger)

