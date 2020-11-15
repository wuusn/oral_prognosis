from glob import glob
import os
from PIL import Image
from multiprocessing import Pool
from itertools import repeat
import numpy as np
import time

src_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/wu_ds' 
base_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease' 

#patch size, magnification, slide window, epi ratio
def atma2patch(im_path, params):
    mag = params['mag']
    psize = params['patch_size']
    swnd = params['slide_window']
    eratio = params['epi_ratio']
    stride = psize*swnd
    stride = int(stride)

    name = im_path.split('/')[-1].replace('.tif', '')
    emask_path = im_path.replace('.tif', '.png')
    label = im_path.split('/')[-2]
    output_dir = f'{base_dir}/patch{mag}x{psize}s{swnd}e{eratio}/{label}/{name}'
    os.makedirs(output_dir, exist_ok=True)

    im = Image.open(im_path)
    scale = 40//mag
    w,h = im.size
    im = im.resize((w//scale, h//scale), Image.BICUBIC)
    w,h = im.size
    emask = Image.open(emask_path)
    emask = emask.resize((w,h), Image.NEAREST)

    for j in range(0,h,stride):
        for i in range(0,w,stride):
            patch=im.crop((i,j,i+psize,j+psize))
            empatch=emask.crop((i,j,i+psize,j+psize))
            emp=np.array(empatch)
            emp = emp/255 if np.max(emp)>1 else emp
            emp = emp.astype(np.uint8)
            if np.sum(emp) > psize*psize*eratio:
                patch.save(f'{output_dir}/{name}_{i}_{j}.png')
    
def one_setting_task(params):
    im_paths = glob(f'{src_dir}/*/*.tif')
    p = Pool(20)
    p.starmap(atma2patch, zip(im_paths, repeat(params)))
    
if __name__ == '__main__':
    start = time.time()
    magnifications = [
                        5,
                        10,
                        #20,
                        #40,
                    ]
    patch_sizes = [
                    #448,
                    224,
                    #112,
                    #56,
                ]
    slide_windows = [
                    #1.0,
                    0.75,
                    #0.5,
                ]
    eratios = [
                0.8,
                #0.7,
            ]
    for m in magnifications:
        for p in patch_sizes:
            for s in slide_windows:
                for e in eratios:
                    params=dict(
                            mag=m,
                            patch_size=p,
                            slide_window=s,
                            epi_ratio=e,
                    )
                    one_setting_task(params)
    #one_setting_task(params)
    end = time.time()
    print('over all time:', (end-start)/60)

    



