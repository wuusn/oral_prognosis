import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import xlsxwriter

class Logger():
    def __init__(self, codename, save_dir):
        self.codename = codename
        self.base_dir = save_dir
        self.save_dir = f'{self.base_dir}/{codename}'
        os.makedirs(self.save_dir, exist_ok=True)
        self.step_losses=[]
        self.step=0
        self.epoch_losses=[]
        self.epoch=0
        self.val_losses=[]
        self.val_count=0 # for logging the results before training
        self.min_val_count=0
        #self.net=None
        writer_dir = f'{self.save_dir}/plots/tensorboard'
        os.makedirs(writer_dir, exist_ok=True)
        self.writer = SummaryWriter(writer_dir)
        self.result=None
        self.min_vloss = float('INF')
        self.checkpoint_dir = f'{self.save_dir}/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def add_validation(self, result, save_model):
        vloss = result['loss']
        self.val_losses.append(vloss)
        self.writer.add_scalar('Loss/val', vloss, self.val_count)
        if vloss < self.min_vloss:
            self.min_vloss = vloss 
            self.min_val_count=self.val_count
            self.result = result

        for m,v in result.items():
            if m=='loss' or m=='net' or m=='slides':
                continue
            else:
                self.writer.add_scalar(f'Val/{m}', v, self.val_count)

        if save_model:
            torch.save(net.to(torch.device('cpu')), f'{self.checkpoint_dir}/{self.val_count}.pth')
            net.to(device)
        self.val_count +=1


    def add_step_loss(self, loss):
        self.step_losses.append(loss)
        self.writer.add_scalar('Loss/train_iteration', loss, self.step)
        self.step+=1

    def add_epoch_loss(self, loss):
        self.epoch_losses.append(loss)
        self.writer.add_scalar('Loss/train_epoch', loss, self.epoch)
        self.epoch +=1

    def save_min_model(self):
        print(f"min-result: loss: {result['loss']}, patch-acc: {result['patch-accuracy']}, slide-auc: {result['slide-auc']}")
        torch.save(self.result['net'].to(torch.device('cpu')), f'{self.checkpoint_dir}/min_{self.min_val_count}.pth')

    def save_plot(self):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, self.epoch), self.epoch_losses, label="train_loss")
        plt.plot(np.arange(0, self.val_count), self.val_losses, label="val_loss")
        plt.title("Training Loss and Validation Loss")
        plt.xlabel("#")
        plt.ylabel("Loss")
        plt.legend(loc="upper left")
        plt.savefig(f'{self.save_dir}/plots/training_plot.png')

    def save_xls(self):
        #save results
        self.save_result_xls()
        #save model-wise metrics global one file
        #self.save_metric_xls()
        #save slide-wise metrics global one file
        #self.save_slide_xls()
    def save_result_xls(self):
        wb = xlsxwriter.Workbook(f'{self.save_dir}/result.xlsx')
        ws = wb.add_worksheet()
        slides = self.result.get('slides')
        i=0
        if slides != None:
            keys = list(slides[0].keys())
            ws.write_row(i,0,keys)
            i+=1
            for s in slides:
                ws.write_row(i,0,list(s.values()))
                i+=1
        ws.write_row(i,0,[' ',' '])
        i+=1
        for k,v in self.result.items():
            if k=='slides'or k=='net':continue
            ws.write_row(i,0,[k,v])
            i+=1
        wb.close()


    def save(self):
        self.save_plot()
        self.save_xls()
        self.save_min_model()

def get_rnd():
    return  np.random.rand(1)[0]

if __name__ == '__main__':
    import time
    codename = time.strftime('%m%d-%H:%M')
    logger = Logger(codename, '/tmp/looggg')
    for i in range(100):
        logger.add_step_loss(get_rnd())
        logger.add_epoch_loss(get_rnd())
        result = dict(
                net=None,
                loss=get_rnd(),
                acc=get_rnd(),
                auc=get_rnd(),
                slides=[{'name':i, 'value':get_rnd()},{'name':i*2, 'value':get_rnd()}]
        )
        logger.add_validation(result, False)
    logger.save_plot()
    logger.save_result_xls()

