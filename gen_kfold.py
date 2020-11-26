from sklearn.model_selection import KFold
import glob
import os
import numpy as np
import random as rn
from openpyxl import load_workbook
import xlsxwriter

np.random.seed(56)
rn.seed(56)

N_Folds=3
Repeat = 10

base_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/wu_ds' 
tar_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/folds_info'

positives = glob.glob(f'{base_dir}/1/*.tif')
negatives = glob.glob(f'{base_dir}/0/*.tif')

positives = [p.split('/')[-1].replace('.tif', '') for p in positives]
negatives = [p.split('/')[-1].replace('.tif', '') for p in negatives]

tmas = {0:negatives, 1:positives}
phases=['train', 'val']

for I in range(0, Repeat):
    #print('round', I)
    oneround={}
    kf = KFold(n_splits=N_Folds, shuffle=True)
    Folds={}
    for i in range(3):
        oneround[i]={}
        oneround[i]['train']=[]
        oneround[i]['val']=[]
    for i in [0,1]:
        Folds[i]=kf.split(tmas[i])
    for g in [0,1]:
        names = tmas[g]
        folds = Folds[g] 
        for i,data in enumerate(folds):
            #print('fold', i)
            for j in range(2):
                phase=data[j]
                for index in phase:
                    name = names[index]
                    #print(name)
                    oneround[i][phases[j]].append(name)
                oneround[i][phases[j]].append('')


    workbook = xlsxwriter.Workbook(f'{tar_dir}/round{I}.xlsx')
    red_format = workbook.add_format()
    red_format.set_bg_color('red')
    green_format = workbook.add_format()
    green_format.set_bg_color('green')
    for i in range(3):
        worksheet = workbook.add_worksheet(f'fold{i}')
        for j in range(2):
            worksheet.write_string(0,j,phases[j])
            color = green_format
            k=1
            for name in oneround[i][phases[j]]:
                if name =='':
                    color = red_format
                    continue
                worksheet.write_string(k,j, name, color)
                k+=1
    workbook.close()
            
                







