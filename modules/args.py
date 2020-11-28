import argparse
import os
import time

def get_args():
    parser = argparse.ArgumentParser(description='Train TMA based classification', \
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nnmodel', type=str, default='vgg16')
    parser.add_argument('--stopby', type=str, default='counting')
    parser.add_argument('--augmentations', type=str, default='flip,simple_color')
    parser.add_argument('--fold-xls-dir', type=str, default='/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/folds_info')
    parser.add_argument('--base-dir', type=str, default='/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/patch20x224s1.0e0.8')
    parser.add_argument('--save-dir', type=str, default='/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/save')
    parser.add_argument('--codename', type=str, default='atest')
    parser.add_argument('--date', type=str, default=time.strftime('%m%d-%H:%M'))
    args = parser.parse_args()
    tmp_dir = f'{args.save_dir}/{args.codename+args.date}'
    os.makedirs(tmp_dir,exist_ok=True)
    with open(f'{tmp_dir}/args.txt', 'w') as f:
        for k,v in args.__dict__.items():
            f.write(f'{k}: {v}\n')
    return args

if __name__ == '__main__':
    args = get_args()
