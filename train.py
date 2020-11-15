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
import matplotlib
matplotlib.use('Agg')

def train(data,net,optimizer,criterion,device,args):
    train_loader, val_loader = data

    train_date = time.strftime('%m%d-%H:%M')
    logger = 
    
    epoch=0
    stopPolicy = getattr(strategy, args.stopby)

    #before training
    stopPolicy.update(vloss)
    logger.update(sth)

    global_step =0
    while stopPolicy.check():
        with tqdm(total=n_train) as pbar:
            for batch in train_loader:
                net.train()
                #
                if global_step % val_step  ==0:
                    net.eval()
                    #try to test a fast way to compute cmatrix
                global_step+=1
        stopPolicy.update(sth)
        logger.update(sth)

        epoch+=1



if __name__ =='__main__': 
    args = get_args()
    data = 
    device =
    net =
    optimizer =
    criterion = 
    train(data,net,optimizer,criterion,device,args)

