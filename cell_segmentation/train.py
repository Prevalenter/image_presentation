import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm

import random
import matplotlib.pyplot as plt
sys.path.append('..')
from loader import MyDataset
from model.unet import UNet

def plot_loss(loss_record, save_path):
    plt.clf()
    plt.plot(loss_record)
    # plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(save_path)

def plot_rmse(train_rmse_list, valid_rmse_list, save_path):
    plt.clf()
    plt.plot(train_rmse_list, label = 'train rmse')
    plt.plot(valid_rmse_list, label = 'valid rmse')
    plt.xlabel('epoch')
    plt.ylabel('Rmse')
    plt.title('Rmse')
    plt.legend()
    plt.savefig(save_path)

def train(args):
    result_path = 'result/%s/'%args.model
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        os.makedirs('%simage'%result_path)
        os.makedirs('%scheckpoint'%result_path)

    train_set = MyDataset('train', args.label_type, 512)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batchsize,
        shuffle=True, 
        num_workers=args.num_workers)

    # device = 'cuda:0'
    device = 'cuda:6' if torch.cuda.device_count()>1 else 'cuda:0'


    out_channels = 1 if args.label_type=='msk' else 2 
    print(out_channels)
    if args.model=='unet':
        print('using unet as model!')
        model = UNet(out_channels=out_channels)
    elif args.model=='deeplab':
        print('using deeplab as model!')
        model = torch.hub.load('pytorch/vision:v0.9.0',
                 'deeplabv3_resnet101', pretrained=False)
    else:
        print('no model!')

    model = model.to(device)
    # model = nn.DataParallel(model)

    img_show = train_set.__getitem__(0)['x']
    img_show = torch.tensor(img_show).to(device).float()[None, :]

    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_list = []
    loss_best = 10
    for epo in tqdm(range(1, args.epochs+1), ascii=True):
        epo_loss = []
        for idx, item in enumerate(train_loader):
            x = item['x'].to(device, dtype=torch.float)
            y = item['y'].to(device, dtype=torch.float)

            optimizer.zero_grad()

            if args.model=='unet':
                pred = model(x)
            elif args.model=='deeplab':
                pred = model(x)['out'][:,0][:,None]

            # print(y.shape, pred.shape)
            loss = criterion(pred, y)
            
            # print(loss.item())
            epo_loss.append(loss.data.item())

            loss.backward()
            optimizer.step()

        epo_loss_mean = np.array(epo_loss).mean()
        # print(epo_loss_mean)
        loss_list.append(epo_loss_mean)
        plot_loss(loss_list, '%simage/loss.png'%result_path)

        with torch.no_grad():
            if args.model=='unet':
                pred = model(img_show.clone())
            elif args.model=='deeplab':
                pred = model(img_show.clone())['out'][:,0][:,None]
            # y = model(img_show)
            # print(img_show.shape)
            if args.label_type=='msk':
                x = img_show[0].cpu().detach().numpy().transpose((1,2,0))
                y = pred[0, 0].cpu().detach().numpy()
            elif args.label_type=='flow':
                x = img_show[0].cpu().detach().numpy().transpose((1,2,0))
                y = pred[0].cpu().detach().numpy().transpose((1,2,0))
                
            plt.subplot(121)
            plt.imshow(x*255)       
            plt.subplot(122)
            plt.imshow(y[:,:,0])  
            plt.savefig('%simage/%d.png'%(result_path, epo))     
            plt.clf()
        #loss
        if epo % 3 ==0:
            torch.save(model, '%scheckpoint/%d.pt'%(result_path, epo))
            if epo_loss_mean < loss_best:
                loss_best = epo_loss_mean
                torch.save(model, '%scheckpoint/best.pt'%(result_path))
            np.save('%sloss.npy'%result_path, np.array(loss_list))

if __name__ == '__main__':
    #python convlstm.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batchsize", default=12, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_batchsize", default=256, type=int)
    parser.add_argument("--multi_cuda", default=True, type=bool)
    # parser.add_argument("--save_path", default='unet', type=str)
    parser.add_argument("--model", default='unet', type=str)
    parser.add_argument("--label_type", default='flow', type=str)
    args = parser.parse_args()


    train(args)