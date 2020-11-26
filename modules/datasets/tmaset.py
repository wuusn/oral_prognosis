from matplotlib import cm
from .patchset import *
import cv2

class TMADataset(PatchDataset):
    # for ONE TMA image
    # iterate patches on ONE TMA
    # only for test yet
    # iterate manually rather than using dataloader!
    # can bring results back to generate heatmap
    # swnd is not supported yet
    def __init__(self, name, ext, mag, psize, base_dir, pad=0,src_mag=40, swnd=1,downby=1,phase='test', augmentations=[]):
        self.name = name
        self.ext = ext
        self.src_mag = src_mag
        self.mag=mag
        self.scale = src_mag//mag
        self.psize=psize
        self.size = self.scale*psize
        self.pad = pad
        self.padsize=self.scale*(self.psize+2*self.pad)
        self.swnd=swnd
        self.downby=downby
        self.stride = int(self.size*swnd)
        self.base_dir = base_dir
        self.phase = phase
        self.augmentations = augmentations
        self.im_path = glob.glob(f'{base_dir}/*/{name}{ext}')[0]
        self.im = Image.open(self.im_path)
        self.label = int(self.im_path.split('/')[-2]) if self.phase=='val' else None
        self.w, self.h = self.im.size
        self.length = math.ceil(self.h/self.stride)*math.ceil(self.w/self.stride)
        self.Left = 0
        self.Top = 0
        self.Right=self.w
        self.Bottom = self.h
        self.om=np.zeros((self.h//self.scale//downby+self.psize, self.w//self.scale//downby+self.psize,3)).astype(np.uint8)
        self.hm=np.zeros((self.h//self.scale//downby+self.psize, self.w//self.scale//downby+self.psize,3)).astype(np.uint8)
        self.pointer=0

    def __len__(self):
        return self.length

    def __getitem__(self,i):
        return self.getimg(i)

    def getimg(self,i):
        jj = i//math.ceil(self.h/self.stride)
        ii = i%math.ceil(self.h/self.stride)
        left = jj * self.size + self.Top
        top = ii * self.size + self.Left
        left_pad = left-self.pad*self.scale
        top_pad = top-self.pad*self.scale
        patch = self.im.crop((left_pad,top_pad,left_pad+self.padsize,top_pad+self.padsize))
        patch = patch.convert(mode='RGB')
        patch = patch.resize((self.psize+2*self.pad, self.psize+2*self.pad), Image.BICUBIC)
        np_patch = np.array(patch).astype(np.uint8)
        tsr_im = self.im2tensor(patch)
        if self.phase=='val':
            return {'image':tsr_im, 'label':self.label, 'ori_im': np_patch}
        else:
            return {'image':tsr_im, 'ori_im': np_patch}

    def overlay(self, i,a,A):
        a = a[self.pad:self.psize+self.pad,self.pad:self.psize+self.pad,:]
        jj = i//math.ceil(self.h/(self.psize)/self.scale)
        ii = i%math.ceil(self.h/(self.psize)/self.scale)
        top = jj * self.psize * self.scale
        left = ii * self.psize * self.scale
        a = cv2.resize(a,(a.shape[1]//self.downby,a.shape[0]//self.downby))
        A[left//self.scale//self.downby:left//self.scale//self.downby+self.psize//self.downby,top//self.scale//self.downby:top//self.scale//self.downby+self.psize//self.downby,:]=a

    def update_origin(self,i):
       np_patch = self.getimg(i)['ori_im']
       self.overlay(i,np_patch,self.om)

    def update_heat(self,i,probability,threshold=.5):
        p=probability
        th = threshold
        ori_patch = self.getimg(i)['ori_im']
        if p>th:
            jet_pred = np.array(cm.jet(p)[:3])*255
            heat = np.ones(ori_patch.shape)*jet_pred
            heat = heat.astype(np.uint8)
            self.overlay(i,heat,self.hm)
        else:
            self.overlay(i,ori_patch,self.hm)

    def save_om(self, tar_dir):
        om = self.om[:self.h//self.scale//self.downby, :self.w//self.scale//self.downby]
        pil_om = Image.fromarray(om)
        pil_om.save(f'{tar_dir}/{self.name}.png')
    def save_hm(self, tar_dir):
        hm = self.hm[:self.h//self.scale//self.downby, :self.w//self.scale//self.downby]
        pil_hm = Image.fromarray(hm)
        pil_hm.save(f'{tar_dir}/{self.name}_heat.png')
    def save(self, tar_dir):
        self.save_om(tar_dir)
        self.save_hm(tar_dir)

class TMAandMaskDataset(TMADataset):
    def __init__(self, name, ext, mext,mag, psize, base_dir, mask_dir, pad=0,src_mag=40, swnd=1,downby=1,phase='test', augmentations=[]):
        super().__init__(name, ext, mag, psize, base_dir, pad,src_mag, swnd,downby,phase, augmentations)
        self.mext=mext
        self.mask_dir = mask_dir
        self.mask_path = self.im_path.replace(ext,mext) # not from mask_dir
        self.mask = Image.open(self.mask_path).convert('L')
        self.mask = self.mask.resize(self.im.size, Image.NEAREST)
    def __len__(self):
        return 0
    def __getitem__(self):
        return None
    def getmask(self,i):
        jj = i//math.ceil(self.h/self.stride)
        ii = i%math.ceil(self.h/self.stride)
        left = jj * self.size + self.Top
        top = ii * self.size + self.Left
        left_pad = left-self.pad*self.scale
        top_pad = top-self.pad*self.scale
        patch = self.mask.crop((left_pad,top_pad,left_pad+self.padsize,top_pad+self.padsize))
        patch = patch.resize((self.psize+2*self.pad, self.psize+2*self.pad), Image.BICUBIC)
        np_patch = np.array(patch).astype(np.uint8)
        return np_patch
    def hasNext(self):
        return self.pointer < self.length
    def next(self):
        im = self.getimg(self.pointer)['ori_im']
        mask = self.getmask(self.pointer)
        self.update_origin(self.pointer)
        self.update_heat(self.pointer,0)
        self.pointer+=1
        return im,mask
    def get_by_enough(self,threshold):
        while self.hasNext():
            im,mask = self.next()
            mask = mask/255 if np.max(mask)>1 else mask
            ratio = np.sum(mask)/(mask.shape[0]**2)
            if ratio>threshold:
                self.update_origin(self.pointer-1)
                return self.getimg(self.pointer-1),self.pointer-1
        return None,None


def get_tmaWiseMaskDatasets(fold_dir,base_dir,mask_dir, r,f,ext, mext,mag, psize, pad=0,src_mag=40, swnd=1,downby=1,phase='val', augmentations=[]):
    names = get_names(fold_dir,r,f,phase)
    datasets = []
    for name in names:
        dataset = TMAandMaskDataset(name,ext,mext,mag,psize,base_dir,mask_dir,pad,src_mag,swnd,downby,phase,augmentations)
        datasets.append(dataset)
    return datasets



if __name__ == '__main__':
    base_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/wu_ds'
    #name = 'TMA 1_006'
    #dataset = TMADataset(name, '.tif', 10, 212, base_dir,pad=6,downby=2)
    #swnd is not supported
    #for i in range(len(dataset)):
    #    #p = np.random.rand(1)[0]
    #    p=0.2
    #    dataset.update_origin(i)
    #    dataset.update_heat(i,p)
    #dataset.save('/tmp')
    #dataset = TMAandMaskDataset(name, '.tif','.png', 10, 212, base_dir,base_dir,pad=6,downby=1)
    fold_xls_dir = '/mnt/md0/_datasets/OralCavity/TMA_arranged/WU/data4disease/folds_info'
    r=0
    f=0
    datasets = get_tmaWiseMaskDatasets(fold_xls_dir,base_dir,base_dir,r,f,'.tif','.png',10,224)
    dataset=datasets[0]
    i=0
    while dataset.hasNext():
        batch,p = dataset.get_by_enough(.5)
        if batch!=None:
            dataset.update_heat(p,1)
            i+=1
    print(i)
    dataset.save_hm('/tmp')
    

