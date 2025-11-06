import numpy as np
import matplotlib.pyplot as plt
import odl
import torch
# from skimage.measure import compare_psnr as skpsnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
import random
from skimage import transform
from odl.contrib import torch as odl_torch
# 读取RAW文件
filename = 'waterBone_proj_512x360.raw'
width = 512
height = 360

# 使用numpy读取数据
data = np.fromfile(filename, dtype=np.float32)
data = data.reshape((height, width))


def radon_op():
    xx = 35.84
    space = odl.uniform_discr([-xx, -xx], [xx, xx], [512, 512], dtype='float32')
    ##_________limited__________
    # angles = np.array(120).astype(int)
    # angle_partition = odl.uniform_partition(0, 2/3* np.pi, angles)
    ##__________sparse___________
    # angles = np.array(360).astype(int)
    angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
    detectors = np.array(512).astype(int)
    detector_partition = odl.uniform_partition(-76.8, 76.8, detectors)
    geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=197.246969, det_radius=220.798)
    operator = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm = odl.operator.power_method_opnorm(operator)
    op_norm = torch.from_numpy(np.array(op_norm * 2 * np.pi)).double().cuda()

    op_layer = odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint = odl_torch.operator.OperatorModule(operator.adjoint)
    fbp = odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9) * np.sqrt(2)
    op_layer_fbp = odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

radon, iradon, fbp, op_norm = radon_op()
# data = np.expand_dims(data, axis=0)
data = torch.from_numpy(data)
img = data.unsqueeze(0).unsqueeze(0)
data = fbp(img)
# 显示图像
import matplotlib.pyplot as plt

# 确保数据为2D数组
data = data.squeeze()  # 去除多余的维度，使其变成 (512, 512) 形状

# 设置图像尺寸为 512x512
plt.figure(figsize=(5.12, 5.12))  # 每英寸100像素，总共512像素

# 显示并保存图像
plt.imshow(data, cmap='gray')
plt.axis('off')  # 去除坐标轴
plt.savefig('output_image.png', format='png', bbox_inches='tight', pad_inches=0, dpi=100)  # 设置dpi为100确保尺寸正确
plt.show()


###此种方式1、灰度变ct---缺少调窗——————归一化
# from skimage.metrics import structural_similarity as skssim
# from skimage.metrics import peak_signal_noise_ratio as skpsnr
#
# from torch.utils.data import Dataset, DataLoader
# import os
# from glob import glob
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from numpy.random import RandomState
# import odl
# import random
# from skimage import transform
# from odl.contrib import torch as odl_torch
# import pydicom
# from sklearn.cluster import k_means
# import scipy.io as sio
# import scipy
#
# import numpy as np
# import skimage
# import cv2
# from skimage.segmentation import find_boundaries
# # from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
# #
# #
# # def SAMAug(tI):
# #     sam = sam_model_registry["vit_b"](checkpoint='./sam_vit_b_01ec64.pth')
# #     mask_generator = SamAutomaticMaskGenerator(sam, crop_nms_thresh=0.5, box_nms_thresh=0.5, pred_iou_thresh=0.5)
# #     masks = mask_generator.generate(tI)
# #     tI = skimage.img_as_float(tI)
# #     SegPrior=np.zeros((tI.shape[0],tI.shape[1]))
# #     BoundaryPrior=np.zeros((tI.shape[0],tI.shape[1]))
# #     for maskindex in range(len(masks)):
# #         thismask=masks[maskindex]['segmentation']
# #         stability_score = masks[maskindex]['stability_score']
# #         thismask_=np.zeros((thismask.shape))
# #         thismask_[np.where(thismask==True)]=1
# #         SegPrior[np.where(thismask_==1)]=SegPrior[np.where(thismask_==1)]+stability_score
# #         BoundaryPrior=BoundaryPrior+find_boundaries(thismask_,mode='thick')
# #         BoundaryPrior[np.where(BoundaryPrior>0)]=1
# #     tI[:,:,1] = tI[:,:,1]+SegPrior
# #     tI[:,:,2] = tI[:,:,2]+BoundaryPrior
# #     return BoundaryPrior
#
#
# class CTSlice_Provider(Dataset):
#     def __init__(self, base_path, poission_level=5e6, gaussian_level=0.05, num_view=96):
#         self.base_path = base_path
#         self.slices_path = glob(os.path.join(self.base_path, '*.IMA'))
#         # self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full = self._radon_transform(num_view=360)
#         # self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr = self._radon_transform(num_view=num_view)
#         # self.poission_level = poission_level
#         # self.gaussian_level = gaussian_level
#         # self.num_view = num_view
#         self.rand_state = RandomState(66)
#
#     def radon_op(self):
#         xx = 200
#         space = odl.uniform_discr([-xx, -xx], [xx, xx], [512, 512], dtype='float32')
#         ##_________limited__________
#         # angles = np.array(120).astype(int)
#         # angle_partition = odl.uniform_partition(0, 2/3* np.pi, angles)
#         ##__________sparse___________
#         # angles = np.array(360).astype(int)
#         angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
#         detectors = np.array(512).astype(int)
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
#     def __getitem__(self, index):
#         radon, iradon, fbp, op_norm = self.radon_op()
#         np.random.seed(120)
#         filename = 'combination_proj_512x360.raw'
#         width = 960
#         height = 768
#
#         # 使用numpy读取数据
#         data = np.fromfile(filename, dtype=np.uint8)
#         data = data.reshape((height, width))
#         data = np.expand_dims(data, axis=0)
#         # slice_path = self.slices_path[index]
#         # ###读取初始图像
#         # dcm = pydicom.read_file(slice_path)
#         # ####灰度值变为ct值
#         # dcm.image = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
#         # data_slice = dcm.image
#         # data_slice = np.array(data_slice).astype(float)
#         # ####归一化
#         # data_slice = (data_slice - np.min(data_slice)) / (np.max(data_slice) - np.min(data_slice))
#         # phantom1 = data_slice###双精度
#         #
#         # phantom = torch.from_numpy(phantom1)
#         # phantom = phantom.unsqueeze(0)
#         # sino = radon(phantom)
#         #
#         # # add Poisson noise
#         # intensityI0 = 5e6
#         # scale_value = torch.from_numpy(np.array(intensityI0).astype(np.float))
#         # normalized_sino = torch.exp(-sino / sino.max())
#         # th_data = np.random.poisson(scale_value * normalized_sino)
#         # sino_noisy = -torch.log(torch.from_numpy(th_data) / scale_value)
#         # sino_noisy = sino_noisy * sino.max()
#         #
#         # # add Gaussian noise
#         # noise_std = [0.05]
#         # noise_std = np.array(noise_std).astype(np.float)
#         # # nx, ny = np.array(120).astype(np.int), np.array(800).astype(np.int) #limited
#         # nx, ny = np.array(360).astype(np.int), np.array(800).astype(np.int) #sparse
#         # noise = noise_std * np.random.randn(nx, ny)
#         # noise = torch.from_numpy(noise)
#         # sino_noisy = sino_noisy + noise
#         #
#         # # #_____________
#         # # #   sparse 120
#         # # sino_noisy[:, 1:361:3, :] = 0
#         # # sino_noisy[:, 2:362:3, :] = 0
#         # # #____________
#         # # _____________
#         # #   sparse 90
#         # sino_noisy[:, 1:361:4, :] = 0
#         # sino_noisy[:, 2:362:4, :] = 0
#         # sino_noisy[:, 3:363:4, :] = 0
#         # ____________
#         # #   sparse 60
#         # sino_noisy[:, 1:361:6, :] = 0
#         # sino_noisy[:, 2:362:6, :] = 0
#         # sino_noisy[:, 3:363:6, :] = 0
#         # sino_noisy[:, 4:364:6, :] = 0
#         # sino_noisy[:, 5:365:6, :] = 0
#         # # ____________
#         # # ____________
#         # #   sparse 180
#         # sino_noisy[:, 1:361:2, :] = 0
#         # ____________
#         fbpu = fbp(data)  # (1.512.512)
#         plt.imshow(fbpu, cmap='gray')
#         plt.show()
#
#
#         #___________
#         #   SAM
#         #___________
#         # image = fbpu
#         # # image = image.squeeze(0)
#         # data1 = np.transpose(image, (1, 2, 0))
#         # data1 = data1.cpu().numpy()
#         # data1 = (data1 * 255).astype(np.uint8)
#         # img = np.zeros([data1.shape[0], data1.shape[1], 3], dtype=np.uint8)
#         # data1 = np.squeeze(data1)
#         # img[:, :, 0] = data1
#         # img[:, :, 1] = data1
#         # img[:, :, 2] = data1
#         # output = SAMAug(img)
#         # imgSAM = output
#         # imgSAM = np.expand_dims(imgSAM, axis=0)
#         # # plt.imshow(imgSAM.squeeze())
#         # # plt.show()
#         return data
#
#
#
#     def __len__(self):
#         return len(self.slices_path)
#
#
# if __name__ == '__main__':
#
#     def imwrite(idx, dir, datalist):
#         for i in range(len(datalist)):
#             file_dir = dir[i] + str(idx) + '.png'
#             plt.imsave(file_dir, datalist[i].data.cpu().numpy().squeeze(), cmap="gray")
#
#     def psnr(img1, img2,datarange):
#         if isinstance(img1, torch.Tensor):
#             img1 = img1.unsqueeze(0)
#             img2 = img2.unsqueeze(0)
#         psnr = skpsnr(img1.cpu().numpy(), img2.cpu().numpy(), data_range=datarange)
#         return psnr
#
#     def aver_psnr(img1, img2,datarange):
#         PSNR = 0
#         assert img1.size() == img2.size()
#         for i in range(img1.size()[0]):
#             for j in range(img1.size()[1]):
#                 PSNR += psnr(img1[i, j:j + 1, ...], img2[i, j:j + 1, ...],datarange)
#         return PSNR / (img1.size()[0] * img1.size()[1])
#     def mkdir(path):
#         folder = os.path.exists(path)
#         if not folder:
#             os.makedirs(path)
#
#
#
#
#
#     img_dir = './image/'
#     print('Reading CT slices Beginning')
#     aapm_dataset = CTSlice_Provider(r'F:\dataset_ct\aapm\train')
#     aapm_dataloader = DataLoader(dataset=aapm_dataset, batch_size=1,shuffle=False)
#     if not os.path.exists('./aapm/90/train'):
#         os.makedirs('./aapm/90/train')
#
#     for index, gtt in enumerate(aapm_dataloader):
#
#         gt, fbp, sam, sino, sino_noisy = gtt[0]
#         npy_img = []  # 用来存放img1和img2
#         npy_img.append(gt.squeeze().numpy())
#
#         print(index)
