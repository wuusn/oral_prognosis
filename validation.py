import torch
import numpy as np
#import medpy
import sklearn.metrics

def val_classification_patch_wise(data_loader, net, device, criterion):
    # patch-wise
    net.eval()
    all_labels = np.array([])
    all_preds = np.array([])
    all_scores = np.array([])
    all_loss=0
    for batch in data_loader:
        x = batch['image'].float().to(device)
        y = batch['label'].to(device)
        size = x.size(0)
        with torch.set_grad_enabled(False):
            outputs = net(x)
            loss = criterion(outputs, y)
            all_loss += loss.item()*size
            outputs = torch.sigmoid(outputs)
            scores, preds = torch.max(outputs, 1)
            all_labels = np.append(all_labels,np.array(y.cpu().numpy()))
            all_preds = np.append(all_preds,np.array(preds.cpu().numpy()))
            all_scores = np.append(all_scores,np.array(scores.cpu().numpy()))
    result={}
    result['net'] = net
    all_count = len(data_loader.dataset) 
    result['loss'] = all_loss/all_count
    #print('image count:',all_count)

    # add whatever metrics you want

    for m in ['accuracy', 'f1', 'precision', 'recall']:
        mfun = getattr(sklearn.metrics, f'{m}_score')
        result[m] = mfun(all_labels, all_preds)

    #for m in ['roc_auc']:
    #    mfun = getattr(sklearn.metrics, f'{m}_score')
    #    result[m] = mfun(all_labels, all_scores)

    return result

def static_val_classification_slide_wise(patch_loaders, net, device, criterion, mode='avg', showWarn=True):
    net.eval()
    all_count=0
    all_results={'slides':[]}
    all_loss=0
    all_preds = np.array([]) # all for slide wise
    all_labels= np.array([])
    all_scores = np.array([])
    patch_preds = np.array([]) # patch wise
    patch_labels= np.array([])
    for data_loader in patch_loaders:
        name = data_loader.dataset.names[0]
        if (len(data_loader.dataset)==0):
            if showWarn:
                print('W: skip',name,'having no image!')
            continue
        slide_label = None
        slide_preds = np.array([])
        slide_scores = np.array([])
        slide_loss=0
        for batch in data_loader:
            x = batch['image'].float().to(device)
            y = batch['label']
            patch_labels = np.append(patch_labels,np.array(y.cpu().numpy()))
            if slide_label==None:
                slide_label=y[0].item()
            y=y.to(device)
            size = x.size(0)
            all_count+=size
            with torch.set_grad_enabled(False):
                outputs = net(x)
                loss = criterion(outputs, y)
                slide_loss += loss.item()*size
                outputs = torch.sigmoid(outputs)
                scores, preds = torch.max(outputs, 1)
                slide_preds = np.append(slide_preds,np.array(preds.cpu().numpy()))
                slide_scores = np.append(slide_scores,np.array(scores.cpu().numpy()))
        slide={}
        slide['name']=name
        all_loss+=slide_loss
        patch_preds = np.append(patch_preds, slide_preds)
        slide['loss'] = slide_loss/len(data_loader.dataset)
        slide['label']=slide_label
        slide_score = np.average(slide_scores) if mode=='avg' else np.maximum(slide_scores)
        slide_pred = 1 if slide_score >.5 else 0
        slide['score'] = slide_score
        slide['pred'] = slide_pred
        all_labels = np.append(all_labels,slide_label)
        all_scores = np.append(all_scores, slide_score)
        all_preds = np.append(all_preds, slide_pred)
        all_results['slides'].append(slide)

    # add whatever metrics you want
    all_results['net'] = net
    #print('image count:',all_count)
    all_loss = all_loss/all_count
    all_results['loss']=all_loss

    for m in ['accuracy', 'f1', 'precision', 'recall']:
        mfun = getattr(sklearn.metrics, f'{m}_score')
        all_results['slide-'+m] = mfun(all_labels, all_preds)
    for m in ['accuracy', 'f1', 'precision', 'recall']:
        mfun = getattr(sklearn.metrics, f'{m}_score')
        all_results['patch-'+m] = mfun(patch_labels, patch_preds)

    for m in ['roc_auc']:
        mfun = getattr(sklearn.metrics, f'{m}_score')
        all_results[m] = mfun(all_labels, all_scores)

    return all_results

def fly_val_classification_slide_wise(slide_datasets, net, device, criterion, threshold,mode='avg', save_dir=None):
    net.eval()
    all_labels = np.array([])
    all_preds = np.array([])
    all_scores = np.array([])
    all_loss=0
    slides = {}
    all_count = 0
    for dataset in slide_datasets:
        slide = {}
        name = dataset.name
        label = dataset.label
        preds = np.array([])
        scores = np.array([])
        slide_loss=0
        count=0
        while dataset.hasNext():
            batch,p=dataset.get_by_enough(threshold)
            if batch==None: continue
            x = batch['image'].float().to(device)
            x = x.unsqueeze(0)
            y = torch.tensor(batch['label']).unsqueeze(0).to(device)

            # add dimention todo
            with torch.set_grad_enabled(False):
                output = net(x)
                loss = criterion(output, y)
                slide_loss += loss.item()
                output= torch.sigmoid(output)
                score,pred = torch.max(output,1)
                score = score.cpu().numpy().squeeze(0)
                pred = pred.cpu().numpy().squeeze(0)
                dataset.update_heat(p,score)
                scores = np.append(scores,np.array(score))
                preds = np.append(preds,np.array(pred))
                count+=1
                all_count+=1
        result={}
        if count==0:
            print('W: skip',name,'having no image!')
            continue
            #slide_score=0
        else:
            slide_score = np.average(scores) if mode=='avg' else np.maximum(scores)
        all_loss+=slide_loss
        slide_loss = slide_loss/count if count!=0 else 0
        result['loss'] = slide_loss
        result['score'] = slide_score
        slide_pred = 1 if slide_score >.5 else 0
        result['pred'] = slide_pred
        all_labels = np.append(all_labels,label)
        all_scores = np.append(all_scores, slide_score)
        all_preds = np.append(all_preds, slide_pred)
        result['label'] = label
        slides[name]=result
    all_loss = all_loss/all_count
    slides['loss'] = all_loss
    # add whatever metrics you want

    for m in ['accuracy', 'f1', 'precision', 'recall']:
        mfun = getattr(sklearn.metrics, f'{m}_score')
        slides[m] = mfun(all_labels, all_preds)
    for m in ['roc_auc']:
        mfun = getattr(sklearn.metrics, f'{m}_score')
        slides[m] = mfun(all_labels, all_scores)

    slides['net']=net
    #print('image count:',all_count)
    return slides

if __name__ == '__main__':
    import torch.nn as nn
    from models import get_vgg16
    from modules import *
    import time
    start = time.time()
    fold_xls_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/folds_info'
    base_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/patch10x224s1.0e0.8'
    tma_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/wu_ds'
    r=0
    f=0
    augments='flip,simple_color'
    bs=32
    val = get_patchLoader(fold_xls_dir,base_dir,r,f,'val',bs,augments)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:1')
    net = get_vgg16().float().to(device)
    result = val_classification_patch_wise(val, net, device,criterion)
    for k,v in result.items():
        if k=='net': continue
        print(k,v)
    end = time.time()
    print('patch-wise time:', (end-start)/60)
    start = time.time()
    tma_loaders = get_slide_based_patchLoaders(fold_xls_dir,base_dir,r,f,'val',bs,augments)
    #net = net.float().to(device)
    slides = static_val_classification_slide_wise(tma_loaders, net, device, criterion, mode='avg')
    for k,v in slides.items():
        if k=='net': continue
        print(k,v)
    end = time.time()
    print('static tma-wise:', (end-start)/60)
    start = time.time()
    slide_datasets = get_tmaWiseMaskDatasets(fold_xls_dir,tma_dir,tma_dir,r,f,'.tif','.png',20,224)
    #net = net.float().to(device)
    slides = fly_val_classification_slide_wise(slide_datasets, net, device, criterion, threshold=0.8)
    for k,v in slides.items():
        if k=='net': continue
        print(k,v)
    end = time.time()
    print('fly tma-wise time:', (end-start)/60)
