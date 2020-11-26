def get_counting(n_train, epoch):
    return StopPolicy(n_train, epoch)

class StopPolicy:
    def __init__(self, n_train, epoch):
        self.epoch = epoch
        self.n_train = n_train
        self.step_loss = None
        self.epoch_losses = []
        self.val_losses=[]
        self.min_val_loss = None
        self.count_epoch=0
        self.count_step=0

    def update_step(self,loss):
        self.count_step +=1
        if self.step_loss == None:
            self.step_loss = loss
        else:
            self.step_loss+=loss

    def update_validation_loss(self,loss):
        self.val_losses.append(loss)
        if self.min_val_loss == None:
            self.min_val_loss = loss
        elif loss < self.min_val_loss:
            self.min_val_loss = loss

    def update_validation(self,sth):
        self.update_validation_loss(sth['loss'])

    def update_epoch(self):
        self.count_epoch+=1
        epoch_loss = self.step_loss/self.n_train
        self.epoch_losses.append(epoch_loss)
        self.step_loss = None
        return epoch_loss

    #def check_validation(self):
    #    return self.count_step%(self.n_train//self.bs)==0

    def check_continue(self):
        return self.count_epoch < self.epoch


    
if __name__ == '__main__':
    from datasets import *
    import torch.nn as nn
    base_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/patch10x224s1.0e0.8'
    r=0
    f=0
    augments='flip,simple_color'
    bs=32
    data = get_loaders(base_dir,r,f,bs,augments)
    train,val=data
    criterion = nn.CrossEntropyLoss()

    # test stop policy
    stopPolicy = get_counting(data, 2, bs)
    while stopPolicy.check_continue():
        for batch in train:
            size = batch['image'].size(0)
            outputs = torch.rand(size,2)
            y = torch.randint(0,2,(size,))
            #outputs = torch.randn(3,5,requires_grad=True)
            #y = torch.empty(3,dtype=torch.long).random_(5)
            loss = criterion(outputs, y)
            stopPolicy.update_step(loss.item()*size)
            if stopPolicy.check_validation():
                print('check validation')
        epoch_loss = stopPolicy.update_epoch()


