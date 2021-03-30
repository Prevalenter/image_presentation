from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import scipy.ndimage as ndimg
from skimage.segmentation import find_boundaries
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import sys
sys.path.append('..')
from loader import MyDataset
from model.unet import UNet
def flow2msk(flow, prob, grad=1.0, area=150, volume=500):
    l = np.linalg.norm(flow, axis=-1)
    flow /= l[:,:,None]; flow[l<grad] = 0
    flow[[0,-1],:,0], flow[:,[0,-1],1] = 0, 0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1,2)
    strides = np.cumprod(flow.shape[::-1])//2
    dn = (dn * strides[-2::-1]).sum(axis=-1)
    rst = np.arange(flow.size//2) + dn
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, minlength=len(rst))
    hist.shape = rst.shape = flow.shape[:2]
    lab, n = ndimg.label(hist, np.ones((3,3)))
    areas = np.bincount(lab.ravel())
    weight = ndimg.sum(hist, lab, np.arange(n+1))
    msk = (areas<area) & (weight>volume)
    lut = np.zeros(n+1, np.int32)
    lut[msk] = np.arange(1, msk.sum()+1)
    mask = lut[lab].ravel()[rst]
    return hist, lut[lab], mask

if __name__ == '__main__':
    train_set = MyDataset('test', 'flow', 512)
    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=False, 
        num_workers=1)

    device = 'cuda:0'

    model = torch.load('pt/best.pt', map_location='cuda:0')
    model  = model.eval()

    with torch.no_grad():
        for item in train_loader:


            x = item['x'].to(device, dtype=torch.float)
            # img_color = x.cpu().detach().numpy()[0].copy().transpose((1,2,0))*255
            pred = model(x).cpu().detach().numpy()[0]
            x = x.cpu().detach().numpy()[0].transpose((1,2,0))*255
            water, core, msk = flow2msk(
                pred.transpose(1,2,0), None, 1.0, 20, 100)

            edge = ~find_boundaries(msk)
            result = x.copy()
            result[edge==0] = 255
            # fig, axes = plt.subplots(2, 2)
            # ax = axes.flatten()
            # print(pred.shape)   
            # ax[0].imshow(x)  
            # ax[1].imshow(pred[0])
            # ax[2].imshow(msk)
            # ax[3].imshow(result)
            # for i in range(4): ax[i].set_axis_off()

            fig, axes = plt.subplots(1, 1)
            axes.imshow(result)
            axes.set_axis_off()

            fig.tight_layout()
            print(item['name'][0][:-4])
            plt.savefig('pred1/%s.jpg'%item['name'][0][:-4])

            # plt.show()
            plt.clf()
