# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from scipy.linalg import sqrtm
import torch.nn.functional as F
from itertools import combinations

def predict(data_loader, encoder, dfc):
    features_1 = []
    labels_1 = []
    features_2 = []
    labels_2 = []
    encoder.eval()
    dfc.eval()
    
    with torch.no_grad():
        for idx, (img, label) in enumerate(data_loader[0]):
            img = img.cuda()
            feat = dfc(encoder(img)[0])
            features_1.append(feat.detach())
            labels_1.append(label)

        for idx, (img, label) in enumerate(data_loader[1]):
            img = img.cuda()

            feat = dfc(encoder(img)[0])
            features_2.append(feat.detach())
            labels_2.append(label)
                
    return torch.cat(features_1).max(1)[1], torch.cat(labels_1).long(), torch.cat(features_2).max(1)[1], torch.cat(labels_2).long()

def predict_RGB(data_loader, encoder, dfc):
    features_1 = []
    labels_1 = []
    features_2 = []
    labels_2 = []
    encoder.eval()
    dfc.eval()
    
    with torch.no_grad():
        for idx, (img, sens, label) in enumerate(data_loader):
            img = img.cuda()
            feat = dfc(encoder(img)[0])
            
            
            features_1.append(feat[sens == 1].detach())
            labels_1.append(label[sens == 1])
            features_2.append(feat[sens == 2].detach())
            labels_2.append(label[sens == 2])
                
    return torch.cat(features_1).max(1)[1], torch.cat(labels_1).long(), torch.cat(features_2).max(1)[1], torch.cat(labels_2).long()

def predict_tab(data_loader, encoder, dfc, num_sens = 2):
    features_1 = []
    labels_1 = []
    features_2 = []
    labels_2 = []
    encoder.eval()
    dfc.eval()
    feature_list = [[] for i in range(num_sens)]
    label_list = [[] for i in range(num_sens)]
    
    with torch.no_grad():
        for idx, (img, sens, label) in enumerate(data_loader):
            img = img.cuda().float()
            sens = sens.view(-1)
            feat = dfc(encoder(img)[0])

        for i in range(num_sens):
            feature_list[i].append(feat[sens == i].detach())
            label_list[i].append(label[sens == i].detach())
            
        for i in range(num_sens):
            feature_list[i] = torch.cat(feature_list[i]).max(1)[1]
            label_list[i] = torch.cat(label_list[i]).long()
#             features_1.append(feat[sens == 0].detach())
#             labels_1.append(label[sens == 0])
#             features_2.append(feat[sens == 1].detach())
#             labels_2.append(label[sens == 1])
                
    return feature_list, label_list
#     return torch.cat(features_1).max(1)[1], torch.cat(labels_1).long(), torch.cat(features_2).max(1)[1], torch.cat(labels_2).long()

def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = max(y_predicted.max(), y_true.max()) + 1  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
#     print(y_predicted)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size

    return reassignment, accuracy


def entropy(input):
    epsilon = 1e-5
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=0)
    return entropy


def balance(predicted, size_0, num_sens = 2, k=10):
    count = torch.zeros((k, num_sens))
    
    for i in range(size_0):
        count[predicted[i], 0] += 1
    for i in range(size_0, predicted.shape[0]):
        count[predicted[i], 1] += 1

    count[count == 0] = 1e-5

    balance_0 = torch.min(count[:, 0] / count[:, 1])
    balance_1 = torch.min(count[:, 1] / count[:, 0])

    en_0 = entropy(count[:, 0] / torch.sum(count[:, 0]))
    en_1 = entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_0, balance_1).numpy(), en_0.numpy(), en_1.numpy()

def balance_list(predicted_list, num_sens = 2, k=10):
    count = torch.zeros((k, num_sens))
    
    for i, predicted in enumerate(predicted_list):
        for j in predicted:
            count[j, i] += 1
        
#     count[predicted[i], 0] += 1
#     for i in range(size_0):
#         count[predicted[i], 0] += 1
#     for i in range(size_0, predicted.shape[0]):
#         count[predicted[i], 1] += 1

    count[count == 0] = 1e-5
    
    balance_list = [(count[:, i] / count[:, i].max()).min() for i in range(num_sens)]
    en_list = [entropy(count[:, i] / torch.sum(count[:, i])).numpy() for i in range(num_sens)]

#     balance_0 = torch.min(count[:, 0] / count[:, 1])
#     balance_1 = torch.min(count[:, 1] / count[:, 0])

#     en_0 = entropy(count[:, 0] / torch.sum(count[:, 0]))
#     en_1 = entropy(count[:, 1] / torch.sum(count[:, 1]))

    return min(balance_list).numpy(), en_list


def calc_FID(data_loader, encoder, dfc, num_clusters = 10, size = 16):
    encoder.eval()
    dfc.eval()
    features_1 = []
    features_2 = []
    
    cluster_features_1 = [[] for i in range(num_clusters)]
    cluster_features_2 = [[] for i in range(num_clusters)]
    
    with torch.no_grad():
        N_1 = 0
        
        for idx, (img, label) in enumerate(data_loader[0]):
            img = img.cuda()
            pred = dfc(encoder(img)[0]).max(1)[1]
            
            img = F.interpolate(img, size = size)
            
            for i in range(num_clusters):
                if sum(pred == i) >= 1:
                    cluster_features_1[i].append(img[pred == i].detach().view(img[pred == i].shape[0], -1))
            N_1 += img.shape[0]
        
        N_2 = 0
        for idx, (img, label) in enumerate(data_loader[1]):
            img = img.cuda()
            pred = dfc(encoder(img)[0]).max(1)[1]
            
            img = F.interpolate(img, size = size)
            
            for i in range(num_clusters):
                if sum(pred == i) >= 1:
                    cluster_features_2[i].append(img[pred == i].detach().view(img[pred == i].shape[0], -1))
                    
            N_2 += img.shape[0]
            
        FFD = calc_FFD2(cluster_features_1, cluster_features_2, num_clusters)
        
        return FFD
    
def calc_FID_RGB(data_loader, encoder, dfc, FD = False, num_clusters = 2, size = 16, zero_pad = True):
    encoder.eval()
    dfc.eval()
    features_1 = []
    features_2 = []

    cluster_features_1 = [[] for i in range(num_clusters)]
    cluster_features_2 = [[] for i in range(num_clusters)]

    with torch.no_grad():
        N_1 = 0
        N_2 = 0

        for idx, (img, sens, label) in enumerate(data_loader):
            img = img.cuda()
            pred = dfc(encoder(img)[0]).max(1)[1]
            img = F.interpolate(img, size = size)

            img_1, img_2 = img[sens==1], img[sens==2]
            pred_1, pred_2 = pred[sens==1], pred[sens==2]

            for i in range(num_clusters):
                if sum(pred_1 == i) >= 1:
                    cluster_features_1[i].append(img_1[pred_1 == i].detach().view(img_1[pred_1 == i].shape[0], -1))
                if sum(pred_2 == i) >= 1:
                    cluster_features_2[i].append(img_2[pred_2 == i].detach().view(img_2[pred_2 == i].shape[0], -1))
            N_1 += img_1.shape[0]
            N_2 += img_2.shape[0]

    if zero_pad:
        FFDC, Z_F = calc_FFD2(cluster_features_1, cluster_features_2, num_clusters)
    else:
        FFDC, Z_F = calc_FFD3(cluster_features_1, cluster_features_2, num_clusters)
        
    if FD:
        FD = calc_FD(cluster_features_1, cluster_features_2, num_clusters, zero_pad)
        
        return FFDC, Z_F, FD
        
    else:
        return FFDC, Z_F


def calc_FID_tab(data_loader, encoder, dfc, num_sens = 2, num_clusters = 2):
    encoder.eval()
    dfc.eval()
#     features_1 = []
#     features_2 = []

    feature_list = [[] for i in range(num_sens)]
    cluster_features_list = [[[] for i in range(num_clusters)] for i in range(num_sens)]
#     cluster_features_2 = [[] for i in range(num_clusters)]

    with torch.no_grad():
        N_1 = 0
        N_2 = 0

        for idx, (img, sens, label ) in enumerate(data_loader):
            img, sens = img.cuda().float(), sens.view(-1)
            pred = dfc(encoder(img)[0]).max(1)[1]

#             img_1, img_2 = img[sens==0], img[sens==1]
#             pred_1, pred_2 = pred[sens==0], pred[sens==1]
            img_list = [img[sens == i] for i in range(num_sens)]
            pred_list = [pred[sens == i] for i in range(num_sens)]

            for j in range(num_sens):
                for i in range(num_clusters):
                    if sum(pred_list[j] == i) >= 1:
                        cluster_features_list[j][i].append(img_list[j][pred_list[j] == i].detach().view(img_list[j][pred_list[j] == i].shape[0], -1))
                    
#             for i in range(num_clusters):
#                 if sum(pred_1 == i) >= 1:
#                     cluster_features_1[i].append(img_1[pred_1 == i].detach().view(img_1[pred_1 == i].shape[0], -1))
#                 if sum(pred_2 == i) >= 1:
#                     cluster_features_2[i].append(img_2[pred_2 == i].detach().view(img_2[pred_2 == i].shape[0], -1))
                    
#             N_1 += img_1.shape[0]
#             N_2 += img_2.shape[0]

    FFD_list = []
    for i, j in list(combinations(range(num_sens), 2)):
        FFD_list.append(calc_FFD2(cluster_features_list[i], cluster_features_list[j], num_clusters)[0])
        
    return max(np.mean(FFD_list, 1))


def calc_FFD(list_1, list_2, num_clusters):
    print([len(i) for i in list_1])
    print([len(i) for i in list_2])
    
    centroids = [torch.cat(list_1[idx] + list_2[idx], 0).mean(0).view(1,-1).cpu().numpy() for idx in range(num_clusters)]

    cluster_features_1 = [torch.cat(cls, 0).cpu().numpy() - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(list_1)]

    cluster_features_2 = [torch.cat(cls, 0).cpu().numpy() - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(list_2)]

    N = [cluster_features_1[idx].shape[0] + cluster_features_2[idx].shape[0] for idx in range(num_clusters)]
    
    mu_1 = np.array([cls.mean(0) if cls is not None else None for cls in cluster_features_1 ])
    mu_2 = np.array([cls.mean(0) if cls is not None else None for cls in cluster_features_2 ])

    C_1 = np.array([cls.sum(0) if cls is not None else None for cls in cluster_features_1 ])
    C_2 = np.array([cls.sum(0) if cls is not None else None for cls in cluster_features_2])
    
    Z = [np.concatenate((cluster_features_1[idx], cluster_features_2[idx])) for idx in range(num_clusters)]
    Z_F = [sum(sum(Z[idx] ** 2 / (N[idx] - 1))) for idx in range(num_clusters)]
    

    C = np.array([C_1[idx] + C_2[idx] if C_1[idx] is not None else None for idx in range(num_clusters)])


    FFD_1 = [2/ (N[idx])**2 * sum((C_1[idx] - C[idx]/2) ** 2) for idx in range(num_clusters) if N[idx] > 0]
    
    s = [ 1/ (N[idx]**2 * (N[idx] - 1)) * (-2 * (C_1[idx] **2).sum() + 2* (C_1[idx] * C).sum() + (3 - N[idx])/2 * (C[idx] ** 2).sum()) \
         for idx in range(num_clusters)]
    
    FFD_upper = [np.sqrt(FFD_1[idx] + s[idx] + Z_F[idx]) for idx in range(num_clusters)]
    FFD_lower = [np.sqrt(FFD_1[idx] + s[idx]) for idx in range(num_clusters)]
    
    return FFD_upper, FFD_lower

def calc_FFD2_list(lists, num_sens, num_clusters):
    centroids = [torch.cat([lists[i][idx] for i in range(num_sens)], 0).mean(0).view(1,-1).cpu().numpy().astype(np.double) for idx in range(num_clusters)]

    cluster_features_list = [[torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(lists[i])] for i in range(num_sens)]

    N = [sum([cluster_features_list[i][idx].shape[0] for i in range(num_sens)]) for idx in range(num_clusters)]
    
    C_list = [np.array([cls.sum(0) for cls in cluster_features_list[i]]) for i in range(num_sens)]
#     C = np.array([C_1[idx] + C_2[idx] for idx in range(num_clusters)])
    
    Z = [np.concatenate([cluster_features_list[i][idx] for i in range(num_sens)]) for idx in range(num_clusters)]
    Z_F = [sum(sum(Z[idx] ** 2 / (N[idx] - 1))) for idx in range(num_clusters)]

    FFD_1 = [sum(((C_1[idx] - C_2[idx]) / N[idx]) ** 2) for idx in range(num_clusters) if N[idx] > 0]
    
    UH_F = [np.sqrt(sum(sum((cluster_features_1[idx] - C_1[idx]/(N[idx])) ** 2))) for idx in range(num_clusters)]
    VH_F = [np.sqrt(sum(sum((cluster_features_2[idx] - C_2[idx]/(N[idx])) ** 2))) for idx in range(num_clusters)]
    
    s = [ 1/ (N[idx] - 1) * (UH_F[idx] - VH_F[idx])**2 for idx in range(num_clusters)]
    
    FFDC = [np.sqrt(FFD_1[idx] + s[idx] + Z_F[idx]) for idx in range(num_clusters)]
#     FFD_lower = [np.sqrt(FFD_1[idx] + s[idx]) for idx in range(num_clusters)]
    
    return FFDC, Z_F
#     return FFD_1, s, Z_F

def calc_FFD2(list_1, list_2, num_clusters):
    
    print([len(i) for i in list_1])
    print([len(i) for i in list_2])
    
    centroids = [torch.cat(list_1[idx] + list_2[idx], 0).mean(0).view(1,-1).cpu().numpy().astype(np.double) for idx in range(num_clusters)]

    cluster_features_1 = [torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(list_1)]

    cluster_features_2 = [torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(list_2)]

    N = [cluster_features_1[idx].shape[0] + cluster_features_2[idx].shape[0] for idx in range(num_clusters)]
    
    C_1 = np.array([cls.sum(0) for cls in cluster_features_1])
    C_2 = np.array([cls.sum(0) for cls in cluster_features_2])
    C = np.array([C_1[idx] + C_2[idx] for idx in range(num_clusters)])
    
    Z = [np.concatenate((cluster_features_1[idx], cluster_features_2[idx])) for idx in range(num_clusters)]
    Z_F = [sum(sum(Z[idx] ** 2 / (N[idx] - 1))) for idx in range(num_clusters)]

    FFD_1 = [sum(((C_1[idx] - C_2[idx]) / N[idx]) ** 2) for idx in range(num_clusters) if N[idx] > 0]
    
    UH_F = [np.sqrt(sum(sum((cluster_features_1[idx] - C_1[idx]/(N[idx])) ** 2))) for idx in range(num_clusters)]
    VH_F = [np.sqrt(sum(sum((cluster_features_2[idx] - C_2[idx]/(N[idx])) ** 2))) for idx in range(num_clusters)]
    
    s = [ 1/ (N[idx] - 1) * (UH_F[idx] - VH_F[idx])**2 for idx in range(num_clusters)]
    
    FFDC = [np.sqrt(FFD_1[idx] + s[idx] + Z_F[idx]) for idx in range(num_clusters)]
#     FFD_lower = [np.sqrt(FFD_1[idx] + s[idx]) for idx in range(num_clusters)]
    
    return FFDC, Z_F
#     return FFD_1, s, Z_F


# def calc_FFD_no_pad(list_1, list_2, num_clusters):
#     centroids = [torch.cat(list_1[idx] + list_2[idx], 0).mean(0).view(1,-1).cpu().numpy().astype(np.double) for idx in range(num_clusters)]

#     cluster_features_1 = [torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else centroids[idx] \
#                       for idx, cls in enumerate(list_1)]

#     cluster_features_2 = [torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else centroids[idx] \
#                       for idx, cls in enumerate(list_2)]

#     N = [cluster_features_1[idx].shape[0] + cluster_features_2[idx].shape[0] for idx in range(num_clusters)]
    
#     mu_1 = np.array([cls.mean(0) for cls in cluster_features_1 ])
#     mu_2 = np.array([cls.mean(0) for cls in cluster_features_2 ])

#     C_1 = np.array([cls.sum(0) for cls in cluster_features_1 ])
#     C_2 = np.array([cls.sum(0) for cls in cluster_features_2])
    
#     Z = [np.concatenate((cluster_features_1[idx], cluster_features_2[idx])) for idx in range(num_clusters)]
#     Z_F = [sum(sum(Z[idx] ** 2 / (N[idx] - 1))) for idx in range(num_clusters)]
    
#     C = np.array([C_1[idx] + C_2[idx] for idx in range(num_clusters)])

#     FFD_1 = [1/ (N[idx])**2 * sum((C_1[idx] - C[idx]) ** 2) for idx in range(num_clusters) if N[idx] > 0]
# #     FFD_1 = [2/ (N[idx])**2 * sum((C_1[idx] - C[idx]) ** 2) for idx in range(num_clusters) if N[idx] > 0]
#     UH_F = [np.sqrt(sum(sum((cluster_features_1[idx] - C_1[idx]/(N[idx])) ** 2))) for idx in range(num_clusters)]
#     VH_F = [np.sqrt(sum(sum((cluster_features_2[idx] - C_2[idx]/(N[idx])) ** 2))) for idx in range(num_clusters)]
    
#     s = [ 1/ (N[idx] - 1) * (UH_F[idx] - VH_F[idx])**2 for idx in range(num_clusters)]
    
#     FFDC = [np.sqrt(FFD_1[idx] + s[idx] + Z_F[idx]) for idx in range(num_clusters)]
# #     FFD_lower = [np.sqrt(FFD_1[idx] + s[idx]) for idx in range(num_clusters)]
    
#     return FFDC, Z_F

def calc_FFD3(list_1, list_2, num_clusters):
    centroids = [torch.cat(list_1[idx] + list_2[idx], 0).mean(0).view(1,-1).cpu().numpy() for idx in range(num_clusters)]

    cluster_features_1 = [torch.cat(cls, 0).cpu().numpy() - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(list_1)]

    cluster_features_2 = [torch.cat(cls, 0).cpu().numpy() - centroids[idx] if len(cls) > 0 else centroids[idx] \
                      for idx, cls in enumerate(list_2)]

    N_1 = [cluster_features_1[idx].shape[0] for idx in range(num_clusters)]
    N_2 = [cluster_features_2[idx].shape[0] for idx in range(num_clusters)]
    N = [cluster_features_1[idx].shape[0] + cluster_features_2[idx].shape[0] for idx in range(num_clusters)]
        
    mu_1 = np.array([cls.mean(0) for cls in cluster_features_1 ])
    mu_2 = np.array([cls.mean(0) for cls in cluster_features_2 ])

    C_1 = np.array([cls.sum(0) for cls in cluster_features_1 ])
    C_2 = np.array([cls.sum(0) for cls in cluster_features_2])
    
    Z = [np.concatenate((cluster_features_1[idx], cluster_features_2[idx])) for idx in range(num_clusters)]
    Z_F = [sum(sum(Z[idx] ** 2 / (np.sqrt(N_1[idx] - 1) * np.sqrt(N_2[idx] - 1)))) for idx in range(num_clusters)]
    
    C = np.array([C_1[idx] + C_2[idx] for idx in range(num_clusters)])

    FFD_1 = [sum((C_1[idx]/N_1[idx] - C_2[idx]/N_2[idx]) ** 2) for idx in range(num_clusters)]
    
    UH_F = [np.sqrt(sum(sum((cluster_features_1[idx] - C_1[idx]/(N_1[idx] + 1e-5 )) ** 2))) for idx in range(num_clusters)]
    VH_F = [np.sqrt(sum(sum((cluster_features_2[idx] - C_2[idx]/(N_2[idx] + 1e-5 )) ** 2))) for idx in range(num_clusters)]
    
    s = [ (1/ np.sqrt(N_1[idx] - 1) * UH_F[idx] - 1/ np.sqrt(N_2[idx] - 1) * VH_F[idx])**2 for idx in range(num_clusters)]
    
    FFD_upper = [np.sqrt(FFD_1[idx] + s[idx] + Z_F[idx]) for idx in range(num_clusters)]
    
#     return FFD_1, s, Z_F
    return FFD_upper, Z_F



def calc_FD(list_1, list_2, num_clusters, zero_pad = True):
    for cls in list_1:
        if len(cls) > 0:
            feature_shape = cls[0].shape[-1]
            break
            
    centroids = [torch.cat(list_1[idx] + list_2[idx], 0).mean(0).view(1,-1).cpu().numpy().astype(np.double) for idx in range(num_clusters)]
    
    #centroid normalize 
    # time complexity.
    
    cluster_features_1 = [torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else np.empty((0, feature_shape))\
                          for idx, cls in enumerate(list_1)]
    cluster_features_2 = [torch.cat(cls, 0).cpu().numpy().astype(np.double) - centroids[idx] if len(cls) > 0 else np.empty((0, feature_shape))\
                          for idx, cls in enumerate(list_2)]
        
    N_1 = [len(i) for i in cluster_features_1]
    N_2 = [len(i) for i in cluster_features_2]
        
    if zero_pad:
        #zero-padding
        cluster_features_1 = [np.concatenate((cluster_features_1[idx], np.zeros((N_2[idx], cluster_features_1[idx].shape[1]))), 0) for idx in range(num_clusters)]
        cluster_features_2 = [np.concatenate((cluster_features_2[idx], np.zeros((N_1[idx], cluster_features_2[idx].shape[1]))), 0) for idx in range(num_clusters)]

    cls_1_mean = [cls.mean(0) for cls in cluster_features_1]
    cls_2_mean = [cls.mean(0) for cls in cluster_features_2]
#     print([len(i) for i in cluster_features_1])
#     print([len(i) for i in cluster_features_2])
#     print(cls_1_mean)
#     print(cls_2_mean)

    cls_1_cov = [cluster_features_1[i] - cls_1_mean[i] for i in range(num_clusters)]
    cls_1_cov = [1/(cluster_features_1[idx].shape[0]-1) * cls_1_cov[idx].T @  cls_1_cov[idx] for idx in range(num_clusters)]
    cls_2_cov = [cluster_features_2[i] - cls_2_mean[i] for i in range(num_clusters)]
    cls_2_cov = [1/(cluster_features_2[idx].shape[0]-1) * cls_2_cov[idx].T @  cls_2_cov[idx] for idx in range(num_clusters)]
    
#     print(cls_1_cov)
#     print(cls_2_cov)

    FD = [sum((cls_1_mean[i] - cls_2_mean[i]) ** 2) + np.trace(cls_1_cov[i] + cls_2_cov[i] -2 * sqrtm(cls_1_cov[i] @ cls_2_cov[i]).real) for i in range(num_clusters)]
        
    return FD

