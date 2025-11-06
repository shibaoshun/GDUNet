import numpy as np
import torch.optim
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import torch
from os.path import join
import time
import torch.utils.data as udata
import logging
from DoubleTightFrames.utilsDnCNN import *
import math
from torch.autograd import Variable
import sympy
import sys
from torch.nn.modules.loss import _Loss
# from skimage.measure.simple_metrics import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import structural_similarity as compare_ssim

def Radial_Line_Sensing_numpy(im, L):
    [h, w] = np.shape(im)
    if 1:
        aperture = (math.pi / 180) * 180
        direction = (math.pi / 180) * 0
    S = LineMaskLimitedAngle_numpy(L, h, aperture, direction)
    SS = np.ravel(S, 'F')
    P = np.where(SS == 1)

    return S, P
def LineMaskLimitedAngle_numpy(L, n, aperture, direction):
    if (math.pi - aperture) > (aperture / L):
        thc = np.linspace(-direction - aperture / 2, -direction + aperture / 2, L)
    else:
        thc = np.linspace(-direction - math.pi / 2, -direction + math.pi / 2 - math.pi / L, L)
    thc = thc % math.pi
    S = np.zeros((n, n))
    for ll in range(L):
        if ((thc[ll] <= math.pi / 4) or (thc[ll] > 3 * math.pi / 4)):
            yr = (np.round(
                (math.tan(thc[ll]) * np.array(range(-n // 2 + 1, n // 2, 1)) + n // 2 + 1).astype(float))).astype(int)
            for nn in range(n - 1):
                S[yr[nn] - 1, nn + 1] = 1
        else:
            xc = (np.round(
                (sympy.cot(thc[ll]) * np.array(range(-n // 2 + 1, n // 2, 1)) + n // 2 + 1).astype(float))).astype(int)
            for nn in range(n - 1):
                S[nn + 1, xc[nn] - 1] = 1
    S = np.fft.ifftshift(S)
    return S
def gen_mask(ori_image, num_radial):  # ori_image:(256,256),生成的mask也为（256，256）
    mask, P = Radial_Line_Sensing_numpy(ori_image, num_radial)
    return mask
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print('Total number of parameters: %d' % num_params)
    return num_params

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])
def batch_PSNR_ssim(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        SSIM += ssim(Iclean[i,i,:,:], Img[i,i,:,:], data_range=1)
    return (PSNR/Img.shape[0]), (SSIM/Img.shape[0])
# def init_logger(argdict):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(level=logging.INFO)
#     fh = logging.FileHandler(join(argdict.out_dir, 'log.txt'), mode='a')
#     formatter = logging.Formatter('%(asctime)s - %(message)s')
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#     logger.info("Arguments: ")
#     for k in argdict.__dict__:
#         logger.info("\t{}: {}".format(k, argdict.__dict__[k]))
#     return logger
class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "result_log_singlecoil_ReBMDual_duandian"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)
        self.terminal = stream
        self.log = open(filename, 'a+')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
class lkxDataset(udata.Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train:
            self.data = np.load("data/train_gt_4000.npy")
        else:
            self.data = np.load("data/val_gt_500.npy")
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        if self.train:
            self.data = np.load("data/train_gt_4000.npy")
        else:
            self.data = np.load("data/val_gt_500.npy")
        x = self.data[index]
        return x

def Myl2_reg_ortho(W):
    # for W in mdl.parameters():
    #     if W.ndimension() < 2:
    #         continue
    #     else:
    cols = W[0].numel()                   #####numel()函数：返回数组中元素的个数121
    cols=min(cols, W[0,0,:,:].numel()) # lqs 121
    w1 = W.view(-1,cols)   # 121 121
    wt = torch.transpose(w1,0,1)   ###这步就是求转置
    m  = torch.matmul(wt,w1)   ## W1t*W1
    ident = torch.eye(cols,cols).cuda()   # 121 * 121的I
    w_tmp = m - ident      ##### W1t-W1 - I
    l2_reg=torch.norm(w_tmp,2)**2    ##### 二范数的平方
    return l2_reg
class my_two_TF_sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum') + lamuda * ||WtW - I||_F^2
    The backward is defined as: input-target
    """
    def __init__(self, size_average=False, reduce=True):
        super(my_two_TF_sum_squared_error, self).__init__(size_average, reduce)

    def forward(self, input, target, W1, lamuda):#, W1, W2,
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        firstterm = nn.functional.mse_loss(input, target, size_average=False, reduce=True)
        # firstterm = nn.functional.mse_loss(input, target, size_average=False, reduce=True)
        secondterm = lamuda * Myl2_reg_ortho(W1)
        # thirdterm=0.01*nn.functional.mse_loss(xrec, target, size_average=False, reduce=True)
        total_loss = firstterm + secondterm

        return total_loss