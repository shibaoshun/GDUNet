import os
from glob import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.random import RandomState
import odl
# from skimage.measure import compare_psnr as skpsnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
import random
from skimage import transform
from odl.contrib import torch as odl_torch
import pydicom
from sklearn.cluster import k_means
import scipy.io as sio
import scipy

from segment_anything import sam_model_registry
# device = args.druevice
# medsam_model = sam_model_registry["vit_b"](checkpoint=r'D:\hlj\ampnet\work_dir\MedSAM\medsam_vit_b.pth')
# medsam_model = medsam_model.cuda()
# medsam_model.eval()
# def image_get_minmax():
#     return 0.0, 1.0
#
# def proj_get_minmax():
#     return 0.0, 200.0
#
# def normalize_torch(data, minmax):
#     data_min, data_max = minmax
#     data = torch.clamp(data, data_min, data_max)  # 限制范围
#     data = (data - data_min) / (data_max - data_min)  # 归一化到 [0, 1]
#     data = data.to(torch.float32)  # 转换为 float32
#     data = data * 255.0  # 缩放到 [0, 255]
#     return data
def image_get_minmax():
    return 0.0, 1.0
def proj_get_minmax():
    return 0.0, 200.0


def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    # data = data * 2.0 - 1.0
    data = data.astype(np.float32)
    data = data * 255.0
    return data


class Phantom2dDatasettest():
    def __init__(self, args, phase, datadir, length, angle):
        self.phase = phase
        self.angles = angle
        self.args = args
        self.base_path = datadir
        # self.sp_file = glob(os.path.join(self.base_path, '*.npy'))
        self.sp_files = glob(os.path.join(self.base_path, '*.npy'))
        # self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full = self.radon_transform(num_view=360)
        # self.radon, self.iradon, self.fbp, self.op_norm = self.radon_transform()
        # self.rand_state = RandomState(66)  #shan
        self.radon, self.iradon, self.fbp, self.op_norm = self.radon_transform_120(num_view=120)


    def __len__(self):

        return len(self.sp_files)

    def radon_transform_120(self, num_view=120):
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.args.img_size[0], self.args.img_size[1]], dtype='float32')
        if num_view == 120 or self.args.mode == 'sparse':
            angle_partition = odl.uniform_partition(0, 2 * np.pi, self.args.sino_size[0])
        elif self.args.mode == 'limited':
            angles = np.array(self.args.sino_size[0]).astype(int)
            angle_partition = odl.uniform_partition(0, 2 / 3 * np.pi, angles)

        detectors = np.array(self.args.sino_size[1]).astype(int)
        detector_partition = odl.uniform_partition(-480, 480, detectors)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600,
                                            det_radius=290)  # FanBeamGeometry
        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        op_norm = odl.operator.power_method_opnorm(operator)
        op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

    def radon_transform(self, num_view=120):
        xx = 200
        space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.args.img_size[0], self.args.img_size[1]], dtype='float32')
        if num_view == 360 or self.args.mode == 'sparse':
            angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
        elif self.args.mode == 'limited':
            angles = np.array(self.args.sino_size[0]).astype(int)
            angle_partition = odl.uniform_partition(0, 2 / 3 * np.pi, angles)

        detectors = np.array(self.args.sino_size[1]).astype(int)
        detector_partition = odl.uniform_partition(-480, 480, detectors)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600,
                                            det_radius=290)  # FanBeamGeometry
        operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

        op_norm = odl.operator.power_method_opnorm(operator)
        op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

        op_layer = odl_torch.operator.OperatorModule(operator)
        op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
        fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
        op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

        return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

    def __getitem__(self, ii):
        # angle = np.random.choice(self.angles)  #shan
        #
        # if 'IMA' in self.sp_file[ii]:
        #     dcm = pydicom.read_file(self.sp_file[ii])
        #     dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        #     data = dcm.image
        #     data = np.array(data).astype(float)
        #     data = transform.resize(data, (self.args.img_size))
        #     data = (data - np.min(data)) / (np.max(data) - np.min(data))
        #
        # data = torch.from_numpy(data)
        # Xgt = data.unsqueeze(0)
        # sino = self.radon(Xgt)  #
        #
        # # -----------
        # # add Poisson noise
        # intensityI0 = self.args.poiss_level
        # scale_value = torch.from_numpy(np.array(intensityI0).astype(np.float))
        # normalized_sino = torch.exp(-sino / sino.max())
        # th_data = np.random.poisson(scale_value * normalized_sino)
        # sino_noisy = -torch.log(torch.from_numpy(th_data) / scale_value)
        # sino_noisy = sino_noisy * sino.max()
        #
        # # add Gaussian noise
        # noise_std = self.args.gauss_level
        # noise_std = np.array(noise_std).astype(np.float)
        # if self.args.mode == 'limited':
        #     nx, ny = np.array(self.args.sino_size[0]).astype(np.int), np.array(self.args.sino_size[1]).astype(np.int)
        # if self.args.mode == 'sparse':
        #     nx, ny = np.array(360), np.array(self.args.sino_size[1]).astype(np.int)
        # noise = noise_std * np.random.randn(nx, ny)
        # noise = torch.from_numpy(noise)
        # sino_noisy = sino_noisy + noise  # Ssv120
        #
        # # -----------
        # if angle == 60:  # 60
        #     sino_noisy[:, 1:361:6, :] = 0
        #     sino_noisy[:, 2:362:6, :] = 0
        #     sino_noisy[:, 3:363:6, :] = 0
        #     sino_noisy[:, 4:364:6, :] = 0
        #     sino_noisy[:, 5:365:6, :] = 0
        #     sino_noisy_360 = sino_noisy
        #     sino_noisy_angle = sino_noisy[:, 0:360:6, :]
        #     Xfbp = self.fbp(sino_noisy_360)
        #
        # if angle == 90:  # 90
        #     sino_noisy[:, 1:361:4, :] = 0
        #     sino_noisy[:, 2:362:4, :] = 0
        #     sino_noisy[:, 3:363:4, :] = 0
        #     sino_noisy_360 = sino_noisy
        #     sino_noisy_angle = sino_noisy[:, 0:360:4, :]
        #     Xfbp = self.fbp(sino_noisy_360)
        # if angle == 120:  # 120
        #     sino_noisy[:, 1:361:3, :] = 0
        #     sino_noisy[:, 2:362:3, :] = 0
        #     sino_noisy_360 = sino_noisy
        #     sino_noisy_angle = sino_noisy[:, 0:360:3, :]
        #     Xfbp = self.fbp(sino_noisy_360)
        #     max_value = torch.max(Xfbp)
        #     min_value = torch.min(Xfbp)
        # if angle == 180:  # 180
        #     sino_noisy[:, 1:361:2, :] = 0
        #     sino_noisy_360 = sino_noisy
        #     sino_noisy_angle = sino_noisy[:, 0:360:2, :]
        #     Xfbp = self.fbp(sino_noisy_360)
        #
        # #angle = self.angles
        # Xgt = Xgt.type(torch.FloatTensor)
        # sino = sino.type(torch.FloatTensor)
        # Xfbp = Xfbp.type(torch.FloatTensor)
        # sino_noisy_360 = sino_noisy_360.type(torch.FloatTensor)
        # sino_noisy_angle = sino_noisy_angle.type(torch.FloatTensor)
        if 'npy' in self.sp_files[ii]:
            # 加载数据
            data = np.load(self.sp_files[ii], allow_pickle=True)
            # 从加载的数据中获取所需的内容
            phantom = data[0]
            fbpu = data[1]
            imgSAM = data[2]
            sino = data[3]
            sino_noisy = data[4]

            # sino[1:361:6, :] = 0
            # sino[2:362:6, :] = 0
            # sino[3:363:6, :] = 0
            # sino[4:364:6, :] = 0
            # sino[5:365:6, :] = 0
            sino[1:361:3, :] = 0
            sino[2:362:3, :] = 0
            # sino[3:363:4, :] = 0

            sino_gt_90 = sino

            if len(fbpu.shape) == 2:
                img_3c = np.repeat(fbpu[:, :, None], 3, axis=-1)
            else:
                img_3c = fbpu
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1)
            )




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
            #
            # print(phantom.max())
            # print(phantom.min())

            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)

            # print(fbpu.max())
            # print(fbpu.min())

            # plt.imshow(imgSAM, 'gray')
            # plt.show()
            imgSAM = torch.from_numpy(imgSAM)
            imgSAM = imgSAM.unsqueeze(0)
            imgSAM = imgSAM.type(torch.FloatTensor)

            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)



        return phantom, fbpu, sino_noisy,sino_gt_90, imgSAM, angle,img_1024_tensor



        # return Xgt, Xfbp, sino, sino_noisy_360, sino_noisy_angle, angle


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
            sino=data[3]
            sino_noisy = data[4]

            sino[1:361:3, :] = 0
            sino[2:362:3, :] = 0
            # sino[3:363:4, :] = 0
            # sino[4:364:6, :] = 0
            # sino[5:365:6, :] = 0
            sino_gt_60 = sino




            if len(fbpu.shape) == 2:
                img_3c = np.repeat(fbpu[:, :, None], 3, axis=-1)
            else:
                img_3c = fbpu
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1)
            )

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

            sino_gt_60 = torch.from_numpy(sino_gt_60)
            sino_gt_60 = sino_gt_60.unsqueeze(0)
            sino_gt_60 = sino_gt_60.type(torch.FloatTensor)

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


        return phantom, fbpu, sino_noisy,sino_gt_60, imgSAM, angle,img_1024_tensor

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
            sino=data[3]
            sino_noisy = data[4]
            sino2 = torch.from_numpy(sino.copy())
            sino2 = sino2.unsqueeze(0)
            sino2 = sino2.type(torch.FloatTensor)

            sino[1:361:3, :] = 0
            sino[2:362:3, :] = 0
            # sino[3:363:4, :] = 0
            # sino[4:364:6, :] = 0
            # sino[5:365:6, :] = 0
            sino_gt_60 = sino


            if len(fbpu.shape) == 2:
                img_3c = np.repeat(fbpu[:, :, None], 3, axis=-1)
            else:
                img_3c = fbpu
            H, W, _ = img_3c.shape
            # %% image preprocessing
            img_1024 = transform.resize(
                img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            img_1024 = (img_1024 - img_1024.min()) / np.clip(
                img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            # convert the shape to (3, H, W)
            img_1024_tensor = (
                torch.tensor(img_1024).float().permute(2, 0, 1)
            )

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
            #
            # phantom = normalize(phantom, proj_get_minmax())
            # fbpu = normalize(fbpu, image_get_minmax())
            # imgSAM = normalize(imgSAM, proj_get_minmax())
            # sino_noisy = normalize(sino_noisy, image_get_minmax())
            # sino_gt_60 = normalize(sino_gt_60, image_get_minmax())


            phantom = torch.from_numpy(phantom)
            phantom = phantom.unsqueeze(0)
            phantom = phantom.type(torch.FloatTensor)



            sino_gt_60 = torch.from_numpy(sino_gt_60)
            sino_gt_60 = sino_gt_60.unsqueeze(0)
            sino_gt_60 = sino_gt_60.type(torch.FloatTensor)

            fbpu = torch.from_numpy(fbpu)
            fbpu = fbpu.unsqueeze(0)
            fbpu = fbpu.type(torch.FloatTensor)

            imgSAM = torch.from_numpy(imgSAM)
            imgSAM = imgSAM.unsqueeze(0)
            imgSAM = imgSAM.type(torch.FloatTensor)
            # imgSAM=(imgSAM).cpu().numpy()
            # # sino_noisy = np.transpose(Tr, (1, 2, 0))  # 把channel那一维放到最后
            # Tr=Tr.reshape(640,641)


            sino_noisy = torch.from_numpy(sino_noisy)
            sino_noisy = sino_noisy.unsqueeze(0)
            sino_noisy = sino_noisy.type(torch.FloatTensor)


        return phantom, fbpu, sino_noisy,sino_gt_60, imgSAM, angle,img_1024_tensor,sino2
# class Phantom2dDataset():
#     def __init__(self, args, phase, datadir):
#         self.phase = phase
#         self.args = args
#         self.base_path = datadir
#         self.sp_file = glob(os.path.join(self.base_path, '*.IMA'))
#         # self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full = self.radon_transform(num_view=360)
#         self.proj, self.back_proj, self.fbp, self.op_norm = self.radon_op()
#
#     def __len__(self):
#
#         return len(self.sp_file)
#
#     # def radon_transform(self, num_view=120):
#     #     xx = 200
#     #     space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.args.img_size[0], self.args.img_size[1]], dtype='float32')
#     #     if num_view == 360 or self.args.mode == 'sparse':
#     #         angle_partition = odl.uniform_partition(0, 2 * np.pi,120)
#     #     elif self.args.mode == 'limited':
#     #         angles = np.array(self.args.sino_size[0]).astype(int)
#     #         angle_partition = odl.uniform_partition(0, 2 / 3 * np.pi, angles)
#     #
#     #     detectors = np.array(self.args.sino_size[1]).astype(int)
#     #     detector_partition = odl.uniform_partition(-480, 480, detectors)
#     #     geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600,
#     #                                         det_radius=290)  # FanBeamGeometry
#     #     operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
#     #
#     #     op_norm = odl.operator.power_method_opnorm(operator)
#     #     op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()
#     #
#     #     op_layer = odl_torch.operator.OperatorModule(operator)
#     #     op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
#     #     fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
#     #     op_layer_fbp = odl_torch.operator.OperatorModule(fbp)
#     #
#     #     return op_layer, op_layer_adjoint, op_layer_fbp, op_norm
#     def radon_op(self):
#         xx = 200
#         space = odl.uniform_discr([-xx, -xx], [xx, xx], [self.args.img_size[0], self.args.img_size[0]], dtype='float32')
#         angles = np.array(self.args.sino_size[0]).astype(int)
#         angle_partition = odl.uniform_partition(0, 2 * np.pi, angles)
#         detectors = np.array(self.args.sino_size[1]).astype(int)
#         detector_partition = odl.uniform_partition(-480, 480, detectors)
#         geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
#         operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
#
#         op_norm = odl.operator.power_method_opnorm(operator)
#         op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()
#
#         op_layer = odl_torch.operator.OperatorModule(operator)
#         op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
#         fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
#         op_layer_fbp = odl_torch.operator.OperatorModule(fbp)
#
#         return op_layer, op_layer_adjoint, op_layer_fbp, op_norm
#
#     def sampleing(self, total_angles, sampling_interval):
#         sampled_angles = total_angles // sampling_interval
#
#         # 创建一个 [90, 360] 的采样矩阵，初始为全零
#         sampling_matrix = torch.zeros((sampled_angles, total_angles))
#
#         # 在对应的采样点上设置 1
#         for i in range(sampled_angles):
#             sampling_matrix[i, i * sampling_interval] = 1
#
#         return sampling_matrix
#
#     def __getitem__(self, ii):
#         '''
#         load training item one by one
#         '''
#         if 'IMA' in self.sp_file[ii]:
#             decide_data_type = 'dcm'
#
#             dcm = pydicom.read_file(self.sp_file[ii])
#             dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
#             data = dcm.image
#             data = np.array(data).astype(float)
#             data = transform.resize(data, (self.args.img_size))
#             phantom = np.rot90(data, 0)
#             phantom = (phantom - np.min(phantom)) / (np.max(phantom) - np.min(phantom))
#
#         phantom = torch.from_numpy(phantom)
#         phantom = phantom.unsqueeze(0)
#         sino = self.proj(phantom)
#
#         # add Poisson noise
#         intensityI0 = self.args.poiss_level
#         scale_value = torch.from_numpy(np.array(intensityI0).astype(float))
#         normalized_sino = torch.exp(-sino / sino.max())
#         th_data = np.random.poisson(scale_value * normalized_sino)
#         sino_noisy = -torch.log(torch.from_numpy(th_data) / scale_value)
#         sino_noisy = sino_noisy * sino.max()
#
#         # add Gaussian noise
#         noise_std = self.args.gauss_level
#         noise_std = np.array(noise_std).astype(float)
#         nx, ny = np.array(self.args.sino_size[0]).astype(int), np.array(self.args.sino_size[1]).astype(int)
#         noise = noise_std * np.random.randn(nx, ny)
#         noise = torch.from_numpy(noise)
#         sino_noisy = sino_noisy + noise
#
#         fbpu = self.fbp(sino_noisy)
#
#         # fbpu=(fbpu-torch.min(fbpu))/(torch.max(fbpu) - torch.min(fbpu))
#
#         return  sino_noisy,fbpu,phantom
