from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import time
import numpy as np
from utils import ssim_loss
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
import os
import matplotlib.pyplot as plt
from head_HM import init_logger_test
from segment_anything import sam_model_registry
# device = args.druevice
# medsam_model = sam_model_registry["vit_b"](checkpoint=r'D:\hlj\ampnet\work_dir\MedSAM\medsam_vit_b.pth')
# medsam_model = medsam_model.cuda()
# medsam_model.eval()
def tohu(X):  # display window as [-175HU, 275HU]
    CT = (X - 0.192) * 1000 / 0.192
    CT_win = CT.clamp_(-600, 600)
    CT_winnorm = (CT_win + 600) / (600 + 600)
    return CT_winnorm
# def tohu(X):           # display window as [-175HU, 275HU]
#     CT = (X - 0.192) * 1000 / 0.192
#     CT_win = CT.clamp_(-175, 275)
#     CT_winnorm = (CT_win +175) / (275+175)
#     return CT_winnorm
def save_image(idx, dir, datalist):
    for i in range(len(datalist)):
        file_dir = dir[i] + str(idx)+'.png'
        plt.imsave(file_dir, datalist[i].data.cpu().numpy().squeeze(), cmap="gray")


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("---  There exsits folder " + path + " !  ---")


class Tester():
    def __init__(self, args, model, test_dset):
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.bat_size = args.tr_batch
        self.tsbat_size = args.ts_batch
        self.test_dset = test_dset
        self.proj, self.back_proj, self.fbp, self.op_norm = self.test_dset.radon_transform_120(num_view=120)


    def test(self):
        #writer = SummaryWriter("./logs_test")
        logger = init_logger_test(self.args)
        self.model = self.model.cuda()
        ckp = torch.load(self.args.test_ckp_dir)
        self.model.load_state_dict(ckp['model'], False)
        self.model.eval()
        DLoadertest = DataLoader(dataset=self.test_dset,  drop_last=True, batch_size=self.tsbat_size,
                                 shuffle=False)
        print('************test dataset length:{}************'.format(len(self.test_dset)))
        t_start = time.time()
        psnr = 0
        psnr1 = 0
        ssim = 0
        ssim1 = 0
        rmse = 0
        rmse1 = 0

        out_dir = self.args.img_dir + '/DDNet/image/'
        mkdir(out_dir)
        input_dir = self.args.img_dir + '/input/image/'
        mkdir(input_dir)
        gt_dir = self.args.img_dir + '/gt/image/'
        mkdir(gt_dir)

        out_dir_hu = self.args.img_dir + '/DDNet_hu/image/'
        mkdir(out_dir_hu)
        input_dir_hu = self.args.img_dir + '/input_hu/image/'
        mkdir(input_dir_hu)
        gt_dir_hu = self.args.img_dir + '/gt_hu/image/'
        mkdir(gt_dir_hu)

        for n_count, data_batch in enumerate(DLoadertest):
            n_count += 1
            bat_num = len(DLoadertest)
            batch_img, batch_u0, sino_noisy,sino_60, imgSAM, angle,img_1024_tensor = [x.cuda() for x in data_batch]
            # print(batch_img.max())
            # print(batch_img.min())

            A = np.zeros((self.bat_size, 1, 360, 360))
            A = torch.from_numpy(A).cuda()
            b = torch.eye(360)
            b = np.asarray(b)
            b[1:361:2, :] = 0
            # b[2:362:3, :] = 0
            # b[3:363:6, :] = 0
            # b[4:364:6, :] = 0
            # b[5:365:6, :] = 0
            b = torch.tensor(b)
            A[:, :, :, :] = b
            A = A.type(torch.FloatTensor).cuda()

            # A1 = np.zeros((self.valbat_size, 1, 360, 360))
            # A1 = torch.from_numpy(A1).cuda()
            # A1[:, :, :, :] = b
            # A1 = A1.type(torch.FloatTensor).cuda()
            # A=A.cpu().numpy()
            # print(A)

            y = sino_noisy.cuda()  # y (1,1,256,256) 复数
            y_input = y  # 归一化之后的y
            # gt = image_train.float().cuda()  # gt (1,1,256,256)
            zf = self.fbp(sino_noisy)
            # plt.imshow(batch_sino.squeeze().cpu().detach().numpy(), cmap='gray')
            # plt.show()

            # zf = torch.fft.ifft2(y_input).cuda() * 256  # 1 1 256 256
            zf_input = zf  # 1 1 256 256

            # zf_max = torch.max(zf)
            # zf_min = torch.min(zf)

            # xu add
            Ax = sino_60.cuda()
            # Ax_input = torch.fft.ifft2(Ax).cuda() * 256
            Ax_input = self.fbp(Ax)
            Ax_input = Ax_input

            # plt.imshow(Ax_input.squeeze().cpu().detach().numpy(), cmap='gray')
            # plt.show()

            max = torch.max(sino_noisy)
            min = torch.min(sino_noisy)

            # with torch.no_grad():
            #     image_embedding = medsam_model.image_encoder(img_1024_tensor)
            #     image_embedding = F.interpolate(
            #         image_embedding,
            #         size=(512, 512),
            #         mode="bilinear",
            #         align_corners=False,
            #     )  # (1, 1, gt.shape)
            # image_embedding = image_embedding[:, :32, :, :]
            # max_value = torch.max(sino_noisy_360)
            # min_value = torch.min(sino_noisy_360)
            # noise = torch.FloatTensor(batch_img.size()).normal_(mean=0, std=6 / 255.)
            # batch_img_n = batch_img + noise.cuda()
            with torch.no_grad():
                # weiying,batch_x_out = self.model(batch_u0, imgSAM,  0.67)
                batch_x_out, W1 = self.model(zf_input, y_input, A, Ax_input, self.radon, self.iradon, val=False)
                # batch_x_out = self.model(batch_img_n,6/255.)
                # batch_x_out = self.model(batch_u0, imgSAM,0.67)
                # max_value = torch.max(batch_x_out)
                # min_value = torch.min(batch_x_out)
                # print("Max value_f:", max_value.item())
                # print("Min value_f:", min_value.item())
                batch_x_out_60 = tohu(torch.clamp(batch_x_out / 2, 0, 0.5))
                batch_img_60 = tohu(torch.clamp(batch_img / 2, 0, 0.5))
                batch_u0_60 = tohu(torch.clamp(batch_u0*2 , 0, 0.5))



            if self.args.test_save_img == True:
                X = [batch_x_out, batch_img, batch_u0]
                dir = [out_dir, gt_dir, input_dir]
                save_image(n_count, dir, X)
                X_60 = [batch_x_out_60, batch_img_60, batch_u0_60]
                dir_60 = [out_dir_hu, gt_dir_hu, input_dir_hu]
                save_image(n_count, dir_60, X_60)

                # X_90 = [val90, Xgt, Xfbp90]
                # dir_90 = [out_dir90, gt_dir90, input_dir90]
                # save_image(n_count, dir_90, X_90)
                #
                # X_120 = [val120, Xgt, Xfbp120]
                # dir_120 = [out_dir120, gt_dir120, input_dir120]
                # save_image(n_count, dir_120, X_120)
                #
                # X_180 = [val180, Xgt, Xfbp180]
                # dir_180 = [out_dir180, gt_dir180, input_dir180]
                # save_image(n_count, dir_180, X_180)
                #
                # Xhu = [Xouthu60, Xouthu90, Xouthu120, Xouthu180]
                # hudir = [out_hudir60, out_hudir90, out_hudir120, out_hudir180]
                # save_image(n_count, hudir, Xhu)


            if n_count % 5 == 0:
                print('[***************************] Epoch {2} of {3}, Batch {0} of {1}'.format(n_count, bat_num, 1, 1))
            psnr_iter = self.aver_psnr(batch_x_out, batch_img)
            ssim_iter = self.aver_ssim(batch_x_out, batch_img)
            rmse_iter = torch.sqrt(torch.mean((batch_x_out - batch_img) ** 2))
            psnr += psnr_iter.item()
            ssim += ssim_iter.item()
            rmse += rmse_iter.item()
            psnr1 += self.aver_psnr(batch_u0, batch_img).item()
            ssim1 += self.aver_ssim(batch_u0, batch_img).item()
            rmse_iter1 = torch.sqrt(torch.mean((batch_u0 - batch_img) ** 2))
            rmse1 += rmse_iter1.item()

            print("\t image:{}  psnr:{:.4f}  ssim:{:.4f}  rmse:{:.4f}  "
                  .format(n_count, psnr_iter, ssim_iter, rmse_iter))
            logger.info("image:{}  psnr:{:.4f}  ssim:{:.4f}  rmse:{:.4f} "
                    .format(n_count, psnr_iter, ssim_iter, rmse_iter))

        print(100 * '*')
        avg_psnr = psnr / n_count
        avg_ssim = ssim / n_count
        avg_rmse = rmse / n_count

        print('\tDDNet mean psnr:{}'.format(avg_psnr))
        print('\tDDNet mean ssim:{}'.format(avg_ssim))
        print('\tDDNet mean rmse:{}'.format(avg_rmse))
        print(100 * '*')
        print('\tFBP mean psnr:{}'.format(psnr1 / n_count))
        print('\tFBP mean ssim:{}'.format(ssim1 / n_count))
        print('\tFBP mean rmse:{}'.format(rmse1 / n_count))
        print(100 * '*')

        logger.info("\t avg_psnr:{:.4f}  avg_ssim:{:.4f}  avg_rmse:{:.4f}"
                    .format(avg_psnr, avg_ssim, avg_rmse))

        t_end = time.time()
        print('\tTest consumes time= %2.2f' % (t_end - t_start))
        print(100 * '*')

    def psnr(self,img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        psnr = skpsnr(img1.cpu().numpy(), img2.cpu().numpy(), data_range=1.0)
        return psnr

    def aver_psnr(self,img1, img2):
        PSNR = 0
        assert img1.size() == img2.size()
        for i in range(img1.size()[0]):
            for j in range(img1.size()[1]):
                PSNR += self.psnr(img1[i, j:j + 1, ...], img2[i, j:j + 1, ...])
        return PSNR / (img1.size()[0] * img1.size()[1])

    def aver_ssim(self,img1, img2):
        '''used in the training'''
        #from skimage.measure import compare_ssim as ski_ssim
        SSIM = 0
        img1 = img1.cpu().numpy().astype(np.float64)
        img2 = img2.cpu().numpy().astype(np.float64)
        for i in range(len(img1)):
            for j in range(img1.shape[1]):
                SSIM += skssim(img1[i, j, ...], img2[i, j, ...], gaussian_weights=True, win_size=11, data_range=1.0,
                               sigma=1.5) #
        return SSIM / (len(img1) * img1.shape[1])
    def radon(self,img):
        if len(img.shape)==4:
            img=img.squeeze(1)
        return self.proj(img).unsqueeze(1)
    def iradon(self,sino):
        if len(sino.shape)==4:
            sino=sino.squeeze(1)
        return self.fbp(sino).unsqueeze(1)
