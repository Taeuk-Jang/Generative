import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision

from dataloader import mnist_usps, mnist_reverse, FaceLandmarksDataset, ImageLoader, ImageDataset
from module import Decoder, Encoder_xya, Decoder_xyz, Encoder_img_xyz, Encoder_img, Decoder_img, Decoder_img_xyz, Classifier
from eval import predict, cluster_accuracy, balance, calc_FID
from utils import set_seed, AverageMeter, target_distribution, aff, inv_lr_scheduler, tsne
import argparse
from PIL import Image

# from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm, trange

def tsne(epoch, trainloader, encoder, save = False):
    with torch.no_grad():
        mnist_lst = list()
        usps_lst = list()
        label_mnist = list()
        label_usps = list()

        for i in range(5):
            x_batch, s_batch, y_batch = iter(trainloader).next()
            x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

            z, mu, logvar = encoder(x_batch)

            mnist_lst.append(z[s_batch.view(-1) == 0])
            usps_lst.append(z[s_batch.view(-1) == 1])
            label_mnist.append(y_batch[s_batch.view(-1) == 0])
            label_usps.append(y_batch[s_batch.view(-1) == 1])

        mnist_lst = torch.cat(mnist_lst)
        usps_lst = torch.cat(usps_lst)
        label_mnist = torch.cat(label_mnist)
        label_usps  = torch.cat(label_usps)

        z = torch.cat([mnist_lst, usps_lst], 0)
        tsne =  TSNE(n_components=2, init='random').fit_transform(z.detach().cpu().numpy())

        plt.figure(figsize = (12,8))
        plt.scatter(tsne[:len(mnist_lst), 0], tsne[:len(mnist_lst), 1], c = label_mnist.cpu().numpy(), label = 'MNIST', cmap=plt.cm.get_cmap("jet"))
        plt.scatter(tsne[len(mnist_lst):, 0], tsne[len(mnist_lst):, 1], c = label_usps.cpu().numpy(), marker = 'x', label = 'USPS', cmap=plt.cm.get_cmap("jet"))
        # plt1.set_color('k')
        # plt2.set_color('k')
        plt1 = plt.scatter([], [], c = 'k', label = 'Group 0', cmap=plt.cm.get_cmap("jet"))
        plt2 = plt.scatter([], [], c = 'k', marker = 'x', label = 'Group 1', cmap=plt.cm.get_cmap("jet"))
        legend = plt.legend(handles = [plt1, plt2], fontsize = 12)
        plt.xticks([])
        plt.yticks([])
        plt.title('tsne z')
        plt.show()
        if save:
            plt.savefig('figure/celeba/whole_space_base_{}.pdf'.format(epoch), ppi = 300, bbox_inches = 'tight')

        

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_size = 224
crop_size = 224

orig_w = 178
orig_h = 218
orig_min_dim = min(orig_w, orig_h)

transform = transforms.Compose([
    transforms.CenterCrop(orig_min_dim),
    transforms.Resize(img_size),
    transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.CenterCrop(orig_min_dim),
    transforms.Resize(img_size),
    transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_set, valid_set, test_set = ImageLoader()
bs = 128

train_data = ImageDataset(train_set, 'Male', 'Attractive', '/data/celebA/CelebA/Img/img_align_celeba', transform)
trainloader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last = True, num_workers = 32)

valid_data = ImageDataset(valid_set, 'Male', 'Attractive', '/data/celebA/CelebA/Img/img_align_celeba', transform_test)
validloader = DataLoader(valid_data, batch_size=bs, shuffle=True, num_workers = 16)

test_data = ImageDataset(test_set, 'Male', 'Attractive', '/data/celebA/CelebA/Img/img_align_celeba', transform_test)
testloader = DataLoader(test_data, batch_size=bs, shuffle=True, num_workers = 16)


kld_weight = 1e-4
samples = 8
epochs = 200

encoder = Encoder_img(latent_variable_size = 1024).cuda()
decoder = Decoder_img(latent_variable_size = 1024).cuda()

optimizer = torch.optim.Adam(encoder.get_parameters() + decoder.get_parameters(), lr = 5e-3)

for epoch in range(epochs):
    iters = tqdm(trainloader)
    step =0
    recons_loss_hist = 0
    kld_loss_hist = 0
    encoder.train()
    decoder.train()
    for x_batch, s_batch, y_batch in iters:
        step += 1
        
        x_batch = x_batch.cuda()
        
        z, mu, logvar = encoder(x_batch)

        recon = decoder(z)

        recons_loss = F.mse_loss(recon, x_batch, reduction = 'mean')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        recons_loss_hist += recons_loss
        kld_loss_hist += kld_loss
        
        iters.set_description('Epoch : {}, recon loss : {:.3f}, KLD loss : {:.3f}'\
                                          .format(epoch, recons_loss_hist/step, kld_loss_hist/step))
        
    if epoch % 3 == 0:
        encoder.eval()
        decoder.eval()
        print('epoch : {}, recon loss : {:.3f}, KLD loss : {:.3f}'.format(epoch, recons_loss, kld_loss))
        recon_1 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0) 
        image_1 = torchvision.utils.make_grid(x_batch[: samples], nrow=samples).cpu().permute(1,2,0) 
        img = torch.cat([recon_1, image_1], 0).detach().numpy()

        plt.figure(figsize = (16,12))
        plt.imshow(img)
        plt.show()
        
        tsne(epoch, trainloader, encoder, True)
        
    torch.save({'state_dict':encoder.state_dict(), 'epoch':epoch}, './save/celeba/encoder_base.pth')
    torch.save({'state_dict':decoder.state_dict(), 'epoch':epoch}, './save/celeba/decoder_base.pth')

        


        
        
        
        