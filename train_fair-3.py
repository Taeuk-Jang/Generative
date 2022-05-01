# Use Projector
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
from module import Classifier, SD_VAE
from eval import predict, cluster_accuracy, balance, calc_FID
from utils import set_seed, AverageMeter, target_distribution, aff, inv_lr_scheduler, tsne, tsne_project
import argparse
from PIL import Image

# from MulticoreTSNE import MulticoreTSNE as TSNE
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm, trange

# based on IntroVAE

if not os.path.exists('./save/celeba/project/'):
    os.makedirs('./save/celeba/project/')
    os.makedirs('./figure/celeba/project/')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img_size = 256
crop_size = 256

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

transform_hq = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    ])

dataname = 'celebahq'
train_set, valid_set = ImageLoader(dataname)
bs = 64

train_data = ImageDataset(train_set, dataname, 'Male', 'Attractive', '/data/celebA/CelebA/Img/img_align_celeba', transform_hq)
trainloader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last = True, num_workers = 32)

valid_data = ImageDataset(valid_set, dataname, 'Male', 'Attractive', '/data/celebA/CelebA/Img/img_align_celeba', transform_hq)
validloader = DataLoader(valid_data, batch_size=bs, shuffle=True, num_workers = 16)


epochs = 10000
samples = 8
beta = 1
lamda_y = 1e2 
lamda_a = 1e1 

sd_vae = nn.DataParallel(SD_VAE()).cuda()

cls_y = nn.DataParallel(Classifier(input_dim = 512)).cuda()
cls_a = nn.DataParallel(Classifier(input_dim = 512)).cuda()


sd_vae.load_state_dict(torch.load('./save/celeba/project/sd_vae.pth')['state_dict'])
cls_y.load_state_dict(torch.load('./save/celeba/project/cls_y.pth')['state_dict'])
cls_a.load_state_dict(torch.load('./save/celeba/project/cls_a.pth')['state_dict'])
epoch = torch.load('./save/celeba/project/cls_a.pth')['epoch']

cls_param_lst = cls_y.module.get_parameters() + cls_a.module.get_parameters()
cls_param_lst[0]['lr_mult'] = 5.0

optimizer_vae = torch.optim.Adam(sd_vae.module.get_parameters(), lr = 1e-6)
optimizer_cls = torch.optim.Adam(cls_param_lst, lr = 1e-5, weight_decay = 1e-5)
# gamma =0.1 with 1000 epochs


sd_vae.eval()
cls_y.train()
cls_a.train()

criterion_bce = torch.nn.BCEWithLogitsLoss(reduction = 'sum')
criterion_ce = torch.nn.CrossEntropyLoss(reduction = 'sum')

# sd_vae.eval()

# cls_y.eval()
# cls_a.eval()

# with torch.no_grad():
#     x_batch, s_batch, y_batch = iter(trainloader).next()
#     x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

#     recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = sd_vae(x_batch)
#     pred_y = cls_y(z_i)
#     pred_a = cls_a(z_r)
        
#     pred_a = torch.sigmoid(pred_a)
#     pred_a[pred_a>=0.5] = 1
#     pred_a[pred_a<0.5] = 0
#     pred_y = torch.sigmoid(pred_y)
#     pred_y[pred_y>=0.5] = 1
#     pred_y[pred_y<0.5] = 0

#     acc_a = (pred_a == s_batch).float().mean()
#     acc_y = (pred_y == y_batch).float().mean()

# #     print('epoch : {}, recon loss : {:.3f}, KLD loss : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
# #                       .format(0, recons_loss, kld_loss, loss_y, loss_a))
#     print('epoch : {}, Acc Y : {:.3f}, Acc A : {:.3f}'.format(0, acc_y, acc_a))

#     recon_1 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0) 
#     image_1 = torchvision.utils.make_grid(x_batch[: samples], nrow=samples).cpu().permute(1,2,0) 

#     z = torch.cat([z_i[0].unsqueeze(0).repeat(bs, 1), z_r], dim = 1)
#     z = sd_vae.module.project(z)
#     recon = sd_vae.module.decode(z)
#     recon_2 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0)

# #     z_i = encoder_i.module.reparameterize(mu_i[0].unsqueeze(0).repeat(128, 1), \
# #                                           logvar_i[0].unsqueeze(0).repeat(128, 1))
#     z = torch.cat([z_i, z_r[0].unsqueeze(0).repeat(bs, 1)], dim = 1)
#     z = sd_vae.module.project(z)
#     recon = sd_vae.module.decode(z)
    
#     recon_3 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0)

#     img = torch.cat([recon_1, recon_2, recon_3, image_1], 0).detach().numpy()


#     plt.figure(figsize = (16,12))
#     plt.imshow(img)
#     plt.savefig('figure/celeba/project/recon_{}.pdf'.format(0), ppi = 300, bbox_inches = 'tight')
# #         plt.show()

#     tsne_project(0, trainloader, sd_vae, True)

for epoch in range(epoch, epochs + 1):
    step =0.
    recons_loss_hist = 0.
    kld_loss_hist = 0.
    loss_y_hist = 0.
    loss_a_hist = 0.
    iters = tqdm(trainloader)
    
    sd_vae.train()
    cls_y.train()
    cls_a.train()

    for x_batch, s_batch, y_batch in iters:
        step += 1
        x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

        recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = sd_vae(x_batch)
        recons_loss = F.mse_loss(recon, x_batch, reduction = 'sum')
        
        kld_loss = 0.
        kld_loss += torch.mean(-0.5 * torch.sum(1 + logvar_i - mu_i ** 2 - logvar_i.exp(), dim = 1), dim = 0)
        kld_loss += beta * torch.mean(-0.5 * torch.sum(1 + logvar_r - mu_r ** 2 - logvar_r.exp(), dim = 1), dim = 0)

        pred_y = cls_y(z_i)
        pred_a = cls_a(z_r)

        loss_y = criterion_bce(pred_y, y_batch)
        loss_a = criterion_bce(pred_a, s_batch)

        loss = recons_loss + kld_loss + lamda_y * loss_y + lamda_a * loss_a

        optimizer_vae.zero_grad()
        optimizer_cls.zero_grad()
        loss.backward()
        optimizer_vae.step()
        optimizer_cls.step()

        recons_loss_hist += recons_loss
        kld_loss_hist += kld_loss
        loss_y_hist += loss_y
        loss_a_hist += loss_a
        
        iters.set_description('epoch : {}, recon loss : {:.3f}, KLD loss : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
                              .format(epoch, recons_loss_hist/step, kld_loss_hist/step, loss_y_hist/step, loss_a_hist/step))
        
    torch.save({'state_dict':sd_vae.state_dict(), 'epoch':epoch}, './save/celeba/project/sd_vae.pth')
    torch.save({'state_dict':cls_y.state_dict(), 'epoch':epoch}, './save/celeba/project/cls_y.pth')
    torch.save({'state_dict':cls_a.state_dict(), 'epoch':epoch}, './save/celeba/project/cls_a.pth')
    
    if epoch % 50 == 0:
        sd_vae.eval()

        cls_y.eval()
        cls_a.eval()
    
        with torch.no_grad():
            pred_a = torch.sigmoid(pred_a)
            pred_a[pred_a>=0.5] = 1
            pred_a[pred_a<0.5] = 0
            pred_y = torch.sigmoid(pred_y)
            pred_y[pred_y>=0.5] = 1
            pred_y[pred_y<0.5] = 0

            acc_a = (pred_a == s_batch).float().mean()
            acc_y = (pred_y == y_batch).float().mean()

            print('epoch : {}, recon loss : {:.3f}, KLD loss : {:.3f}, Y loss : {:.3f}, A loss : {:.3f}'\
                              .format(epoch, recons_loss, kld_loss, loss_y, loss_a))
            print('epoch : {}, Acc Y : {:.3f}, Acc A : {:.3f}'.format(epoch, acc_y, acc_a))

            recon_1 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0) 
            image_1 = torchvision.utils.make_grid(x_batch[: samples], nrow=samples).cpu().permute(1,2,0) 

            z = torch.cat([z_i[0].unsqueeze(0).repeat(bs, 1), z_r], dim = 1)
            z = sd_vae.module.project(z)
            recon = sd_vae.module.decode(z)
            recon_2 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0)

        #     z_i = encoder_i.module.reparameterize(mu_i[0].unsqueeze(0).repeat(128, 1), \
        #                                           logvar_i[0].unsqueeze(0).repeat(128, 1))
            z = torch.cat([z_i, z_r[0].unsqueeze(0).repeat(bs, 1)], dim = 1)
            z = sd_vae.module.project(z)
            recon = sd_vae.module.decode(z)

            recon_3 = torchvision.utils.make_grid(recon[: samples], nrow=samples).cpu().permute(1,2,0)

            img = torch.cat([recon_1, recon_2, recon_3, image_1], 0).detach().numpy()


            plt.figure(figsize = (16,12))
            plt.imshow(img)
            plt.savefig('figure/celeba/project/recon_{}.pdf'.format(epoch), ppi = 300, bbox_inches = 'tight')
    #         plt.show()

            tsne_project(epoch, trainloader, sd_vae, True)
        

