from skimage.io import imread, imsave
from skimage.filters import  sobel
from skimage.transform import resize
import yaml
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
# import cv2

os.environ['KMP_DUPLICATE_LIB_OK']='True'
root = '../data/'

def msk_preproceess(label):
    label = (label>0)*1
    return label[None]

def flow_preproceess(label):
    return label.transpose((2,0,1))


class MyDataset(Dataset):
    def __init__(self, set_type='train', label_type='msk', resolution=512):
        self.root = 'data/%s/img/'%set_type
        self.label_type = label_type
        # msk or flow
        self.label_root = 'data/%s/%s/'%(set_type, label_type)

        self.path_list = os.listdir(self.root)
        self.resolution = resolution

        print('len of set is: ', len(self.path_list))
    def __len__(self):
        list_len = len(self.path_list)
        return list_len

    def __getitem__(self, idx):
        img_path = '%s%s'%(self.root, self.path_list[idx])
        label_path = '%s%s.npy'%(self.label_root, self.path_list[idx].split('.')[0])

        img = imread(img_path)
        # label = imread(label_path)
        label = np.load(label_path)

        img = resize(img, (self.resolution, self.resolution))
        label = resize(label, (self.resolution, self.resolution))

        if self.label_type=='msk': y = msk_preproceess(label)
        elif self.label_type=='flow': y = flow_preproceess(label)

        return {'x':img.transpose((2,0,1))/255,
                 'y':y,
                 'name': self.path_list[idx]}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    train_set = MyDataset('train', 'flow')
    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True, 
        num_workers=4)   

    # item = train_set.__getitem__(10)

    # x = item['x']
    # y = item['y']
    # print(x.shape, y.shape)

    for item in train_loader:
        x = item['x']
        y = item['y']
        print(x.shape, y.shape)

        plt.imshow(y[0,1])
        plt.show()
    #     print(i[0].shape)    
    # for i in range(len(train_set.path_list)):
    #     item = train_set.__getitem__(i)
    #     if item['x'].shape[1:]!=item['y'].shape[1:]:
    #         print(item['x'].shape, item['y'].shape)
        # plt.imshow(item[0].transpose((1,2,0)))
        # # plt.imshow(item[1].transpose((1,2,0)))
        # plt.show()







