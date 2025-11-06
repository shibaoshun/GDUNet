import os
from glob import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.random import RandomState
import odl
from skimage.measure import compare_psnr as skpsnr
import random
from skimage import transform
from odl.contrib import torch as odl_torch
import pydicom
from sklearn.cluster import k_means
import scipy.io as sio
import scipy

class Phantom2dDatasettest():
    def __init__(self, args, phase, datadir, length, angle):
        self.phase = phase
        self.angles = angle
        self.length = length
        self.img_size = args.img_size
        # self.sino_size = args.sino_size
        self.args = args
        self.base_path = datadir
        self.rand_state = RandomState(66)


    def __len__(self):
        return self.length

    def __getitem__(self, ii):
        # np.random.seed(120)
        # 随机一个角度
        angle = np.random.choice(self.angles)
        # 构建每个角度对应的文件路径
        #file_path = os.path.join(self.base_path, str(angle))
        sp_files = glob(os.path.join(self.base_path, '*.npy'))
        # 随机选择一个文件
        if 'npy' in sp_files[ii]:
            # 加载数据
            data = np.load(sp_files[ii], allow_pickle=True)
            # 从加载的数据中获取所需的内容
            phantom = data[0]
            fbpu = data[1]
            imgSAM = data[2]
            sino_noisy = data[4]

            phantom = torch.from_numpy(phantom)
            phantom = phantom.unsqueeze(0)
            phantom = phantom.type(torch.FloatTensor)

            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)

            imgSAM = torch.from_numpy(imgSAM)
            imgSAM = imgSAM.unsqueeze(0)
            imgSAM = imgSAM.type(torch.FloatTensor)

            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)

            # fbpu = np.transpose(fbpu, (1, 2, 0))  # 把channel那一维放到最后
            # plt.imshow(fbpu, 'gray')
            # plt.show()
            # imgSAM = np.transpose(imgSAM, (1, 2, 0))  # 把channel那一维放到最后
            # plt.imshow(imgSAM, 'gray')
            # plt.show()
            # sino_noisy = np.transpose(sino_noisy, (1, 2, 0))  # 把channel那一维放到最后
            # plt.imshow(sino_noisy, 'gray')
            # plt.show()
            # phantom = np.transpose(phantom, (1, 2, 0))  # 把channel那一维放到最后
            # plt.imshow(phantom, 'gray')
            # plt.show()


        return phantom, fbpu, sino_noisy, imgSAM, angle

class Phantom2dDatasetval1():
    def __init__(self, args, phase, datadir, length, angle):
        self.phase = phase
        self.angles = angle
        self.length = length
        self.img_size = args.img_size
        # self.sino_size = args.sino_size
        self.args = args
        self.base_path = datadir
        self.rand_state = RandomState(66)
        # self.file_path = os.path.join(self.base_path)
        self.sp_files = glob(os.path.join(self.base_path, '*.npy'))


    def __len__(self):
        return self.length

    def __getitem__(self, ii):
        # np.random.seed(120)
        # 随机一个角度
        # angle = np.random.choice(self.angles)
        # 每个角度被选择的概率
        # probabilities = [1 / len(self.angles)] * len(self.angles)
        # # 循环100次选择角度
        # angle = np.random.choice(self.angles, size=self.length, p=probabilities)
        # angle = np.random.choice(self.angles)
        # 构建每个角度对应的文件路径
        # file_path = os.path.join(self.base_path, 'val')
        # sp_files = glob(os.path.join(file_path, '*.npy'))
        # 随机选择一个文件
        if 'npy' in self.sp_files[ii]:
            # 加载数据
            data = np.load(self.sp_files[ii], allow_pickle=True)
            # 从加载的数据中获取所需的内容
            phantom = data[0]
            fbpu = data[1]
            imgSAM = data[2]
            sino_noisy = data[4]

            # 计算行全为0的个数
            zero_rows_count = np.sum(np.all(sino_noisy == 0, axis=1))

            # 计算行数
            total_rows = sino_noisy.shape[0]

            # 判断条件是否成立
            # is_condition_met = (total_rows - zero_rows_count) / total_rows

            # 使用np.isclose进行浮点数比较
            if np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 6):
                angle = int(60)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 4):
                angle = int(90)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 3):
                angle = int(120)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 2):
                angle = int(180)
            else:
                # 处理未匹配到任何条件的情况
                angle = None

            phantom = torch.from_numpy(phantom)
            phantom = phantom.unsqueeze(0)
            phantom = phantom.type(torch.FloatTensor)

            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)

            imgSAM = torch.from_numpy(imgSAM)
            imgSAM = imgSAM.unsqueeze(0)
            imgSAM = imgSAM.type(torch.FloatTensor)

            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)


        return phantom, fbpu, sino_noisy, imgSAM, angle

class Phantom2dDatasettrain1():
    def __init__(self, args, phase, datadir, length, angle):
        self.phase = phase
        self.angles = angle
        self.length = length
        self.img_size = args.img_size
        # self.sino_size = args.sino_size
        self.args = args
        self.base_path = datadir
        self.rand_state = RandomState(66)
        # self.file_path = os.path.join(self.base_path)
        self.sp_files = glob(os.path.join(self.base_path, '*.npy'))


    def __len__(self):
        return self.length

    def __getitem__(self, ii):
        # np.random.seed(120)
        # 随机一个角度
        # angle = np.random.choice(self.angles)
        # 每个角度被选择的概率
        # probabilities = [1 / len(self.angles)] * len(self.angles)
        # # 循环100次选择角度
        # angle = np.random.choice(self.angles, size=self.length, p=probabilities)
        # angle = np.random.choice(self.angles)
        # 构建每个角度对应的文件路径
        # file_path = os.path.join(self.base_path, 'val')
        # sp_files = glob(os.path.join(file_path, '*.npy'))
        # 随机选择一个文件
        if 'npy' in self.sp_files[ii]:
            # 加载数据
            data = np.load(self.sp_files[ii], allow_pickle=True)
            # 从加载的数据中获取所需的内容
            phantom = data[0]
            fbpu = data[1]
            imgSAM = data[2]
            sino_noisy = data[4]

            # 计算行全为0的个数
            zero_rows_count = np.sum(np.all(sino_noisy == 0, axis=1))

            # 计算行数
            total_rows = sino_noisy.shape[0]

            # 判断条件是否成立
            # is_condition_met = (total_rows - zero_rows_count) / total_rows

            # 使用np.isclose进行浮点数比较
            if np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 6):
                angle = int(60)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 4):
                angle = int(90)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 3):
                angle = int(120)
            elif np.isclose((total_rows - zero_rows_count) / total_rows, 1 / 2):
                angle = int(180)
            else:
                # 处理未匹配到任何条件的情况
                angle = None

            phantom = torch.from_numpy(phantom)
            phantom = phantom.unsqueeze(0)
            phantom = phantom.type(torch.FloatTensor)

            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)

            imgSAM = torch.from_numpy(imgSAM)
            imgSAM = imgSAM.unsqueeze(0)
            imgSAM = imgSAM.type(torch.FloatTensor)

            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)


        return phantom, fbpu, sino_noisy, imgSAM, angle
