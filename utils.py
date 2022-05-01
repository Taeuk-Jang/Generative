# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import random
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def tsne(epoch, trainloader, encoder_i, encoder_r, save = False):
    with torch.no_grad():
        mnist_lst = list()
        usps_lst = list()
        label_mnist = list()
        label_usps = list()
        mnist_z_r_lst = list()
        usps_z_r_lst = list()
        mnist_z_i_lst = list()
        usps_z_i_lst = list()

        for i in range(5):
            x_batch, s_batch, y_batch = iter(trainloader).next()

            x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

            z_i, mu_i, logvar_i = encoder_i(x_batch)
            z_r, mu_r, logvar_r = encoder_r(x_batch)
            z = torch.cat([z_i, z_r], dim = 1)

            mnist_lst.append(z[s_batch.view(-1) == 0])
            usps_lst.append(z[s_batch.view(-1) == 1])
            mnist_z_r_lst.append(z_r[s_batch.view(-1) == 0])
            usps_z_r_lst.append(z_r[s_batch.view(-1) == 1])
            mnist_z_i_lst.append(z_i[s_batch.view(-1) == 0])
            usps_z_i_lst.append(z_i[s_batch.view(-1) == 1])

            label_mnist.append(y_batch[s_batch.view(-1) == 0])
            label_usps.append(y_batch[s_batch.view(-1) == 1])

        mnist_lst = torch.cat(mnist_lst)
        usps_lst = torch.cat(usps_lst)
        label_mnist = torch.cat(label_mnist)
        label_usps  = torch.cat(label_usps)
        mnist_z_r_lst = torch.cat(mnist_z_r_lst)
        usps_z_r_lst = torch.cat(usps_z_r_lst)
        mnist_z_i_lst = torch.cat(mnist_z_i_lst)
        usps_z_i_lst = torch.cat(usps_z_i_lst)

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
#         plt.show()
        if save:
            plt.savefig('figure/celeba/whole_space_{}.pdf'.format(epoch), bbox_inches = 'tight')

        z = torch.cat([mnist_z_r_lst, usps_z_r_lst], 0)
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
        plt.title('tsne z relevant')
#         plt.show()
        if save:
            plt.savefig('figure/celeba/sens_relevant_{}.pdf'.format(epoch), bbox_inches = 'tight')

        z = torch.cat([mnist_z_i_lst, usps_z_i_lst], 0)
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
        plt.title('tsne z irrelevant')
#         plt.show()
        if save:
            plt.savefig('figure/celeba/sens_irrelevant_{}.pdf'.format(epoch), bbox_inches = 'tight')


def tsne_project(epoch, trainloader, sd_vae, save = False):
    with torch.no_grad():
        mnist_lst = list()
        usps_lst = list()
        label_mnist = list()
        label_usps = list()
        mnist_z_r_lst = list()
        usps_z_r_lst = list()
        mnist_z_i_lst = list()
        usps_z_i_lst = list()

        for i in range(5):
            x_batch, s_batch, y_batch = iter(trainloader).next()

            x_batch, s_batch, y_batch = x_batch.cuda(), s_batch.cuda().float().view(-1,1), y_batch.cuda().float().view(-1,1)

            recon, z_i, mu_i, logvar_i, z_r, mu_r, logvar_r = sd_vae(x_batch)
            z = torch.cat([z_i, z_r], dim = 1)

            mnist_lst.append(z[s_batch.view(-1) == 0])
            usps_lst.append(z[s_batch.view(-1) == 1])
            mnist_z_r_lst.append(z_r[s_batch.view(-1) == 0])
            usps_z_r_lst.append(z_r[s_batch.view(-1) == 1])
            mnist_z_i_lst.append(z_i[s_batch.view(-1) == 0])
            usps_z_i_lst.append(z_i[s_batch.view(-1) == 1])

            label_mnist.append(y_batch[s_batch.view(-1) == 0])
            label_usps.append(y_batch[s_batch.view(-1) == 1])

        mnist_lst = torch.cat(mnist_lst)
        usps_lst = torch.cat(usps_lst)
        label_mnist = torch.cat(label_mnist)
        label_usps  = torch.cat(label_usps)
        mnist_z_r_lst = torch.cat(mnist_z_r_lst)
        usps_z_r_lst = torch.cat(usps_z_r_lst)
        mnist_z_i_lst = torch.cat(mnist_z_i_lst)
        usps_z_i_lst = torch.cat(usps_z_i_lst)

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
#         plt.show()
        if save:
            plt.savefig('figure/celeba/project/whole_space_{}.pdf'.format(epoch), bbox_inches = 'tight')

        z = torch.cat([mnist_z_r_lst, usps_z_r_lst], 0)
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
        plt.title('tsne z relevant')
#         plt.show()
        if save:
            plt.savefig('figure/celeba/project/sens_relevant_{}.pdf'.format(epoch), bbox_inches = 'tight')

        z = torch.cat([mnist_z_i_lst, usps_z_i_lst], 0)
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
        plt.title('tsne z irrelevant')
#         plt.show()
        if save:
            plt.savefig('figure/celeba/project/sens_irrelevant_{}.pdf'.format(epoch), bbox_inches = 'tight')

        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(layer):
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv2d") != -1 or layer_name.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(layer.weight)
    elif layer_name.find("BatchNorm") != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
    elif layer_name.find("Linear") != -1:
        nn.init.xavier_normal_(layer.weight)


def inv_lr_scheduler(optimizer, lr, iter, max_iter, gamma=10, power=0.75):
    learning_rate = lr * (1 + gamma * (float(iter) / float(max_iter))) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group["lr_mult"]
        i += 1

    return optimizer


def target_distribution(batch):
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def aff(input):
    return torch.mm(input, torch.transpose(input, dim0=0, dim1=1))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
