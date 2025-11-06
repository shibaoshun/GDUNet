import argparse
from numpy import *
from scipy.io import savemat

from Model import *
import torchvision.transforms as transforms
import cv2
import scipy.io as scio

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设备号 默认为0

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print('Total number of parameters: %d' % num_params)
    return num_params

def main(opt):
    L = np.array([15])
    for index in range(len(L)):
        num_radial = L[index]
        torch.manual_seed(0)
        sigma = 0.1
        Pcount_psnr = []
        Pcount_ssim = []
        ##### 加载训练好的模型
        model = network_sct(layers_num=opt.depths, sigma=opt.sigma, beta=opt.beta, gamma=opt.gamma).cuda()

        num_params = print_network(model)
        print('Total number of parameters: %d' % num_params)

        # model.load_state_dict(torch.load(os.path.join(opt.out_dir, 'knee2000_radial.model')))  ###  导入训练好的模型
        model.load_state_dict(torch.load(os.path.join(opt.out_dir, 'knee4000_radial.model')))  ###  导入训练好的模型
        model.eval()
        path = 'data/final_test_knee'  #####  测试图片的路径
        path_list = os.listdir(path)
        for filename in path_list:
            print(filename)
            file = os.path.join(path, filename)
            ori_image = cv2.imread(file, 0)  ## 图像 256 256   cv2.imread后面读的必须是英文路径，不能有中文，否则为none
            loader = transforms.Compose([transforms.ToTensor()])  # ToTensor:H W C 变成 C H W  且[0 1]  Compose:整合步骤
            image_tensor0 = loader(ori_image)  # 1 256 256 (0-1）
            image_test = image_tensor0.unsqueeze(0)  # 1 1 256 256
            # mask
            val_gt_use_for_gen_mask_noise = image_test.squeeze(1).numpy() # 1 256 256
            [m, u, v] = np.shape(val_gt_use_for_gen_mask_noise)
            # mask = np.zeros([m, u, v])
            # for i in range(0, m):
            #     arr = val_gt_use_for_gen_mask_noise[i, :, :]
            #     mask0 = gen_mask(arr, num_radial)
            #     mask[i, :, :] = mask0  #####1,256,256
            # mask = torch.tensor(mask).unsqueeze(-1).float()  ####tensor 1,256,256,1
            # mask_val = mask.permute(0, 3, 1, 2).cuda()# 1 1 256 256

            for index in range(len(L)):
                # logger = init_logger(args)
                num_radial = L[index]
                if num_radial == 15:
                    data = scio.loadmat('data/final_mask\Q_cartesian_0.05.mat')
                    mask = np.transpose(data['mask'][:])
                    mask = np.fft.ifftshift(mask)
                    mask = torch.FloatTensor(mask).cuda()  # 1在四周的radial # 256 256
                    mask_val = mask.unsqueeze(0).unsqueeze(0)

                elif num_radial == 25:
                    data = scio.loadmat('data/final_mask\Q_cartesian_0.1.mat')
                    mask = np.transpose(data['mask'][:])
                    mask = np.fft.ifftshift(mask)
                    mask = torch.FloatTensor(mask).cuda()  # 1在四周的radial # 256 256
                    mask_val = mask.unsqueeze(0).unsqueeze(0)

                elif num_radial == 55:
                    data = scio.loadmat('data/final_mask\Q_cartesian_0.2.mat')
                    mask = np.transpose(data['mask'][:])
                    mask = np.fft.ifftshift(mask)
                    mask = torch.FloatTensor(mask).cuda()  # 1在四周的radial # 256 256
                    mask_val = mask.unsqueeze(0).unsqueeze(0)

                elif num_radial == 85:
                    data = scio.loadmat('data/final_mask\Q_cartesian_0.3.mat')
                    mask = np.transpose(data['mask'][:])
                    mask = np.fft.ifftshift(mask)
                    mask = torch.FloatTensor(mask).cuda()  # 1在四周的radial # 256 256
                    mask_val = mask.unsqueeze(0).unsqueeze(0)

            # noise
            noise = sigma/255 * torch.randn(256, 256) + sigma /255 * torch.randn(256, 256) * (1.j)
            noise_val = noise.unsqueeze(0).unsqueeze(1).cuda()  # 1 1 256 256
            #
            con_arr_torch = torch.tensor(val_gt_use_for_gen_mask_noise).unsqueeze(-1).float()  # 1 256 256 1
            con_arr = con_arr_torch.permute(0, 3, 1, 2)  # 1 1 256 256
            con_arr = torch.fft.fft2(con_arr).cuda() / 256  # 1 1 256 256
            con_arr_tensor_fft_mask = con_arr * mask_val
            con_arr_tensor_fft_mask_noise = con_arr_tensor_fft_mask + noise_val  # tensor 1 1 256 256
            val_xdata = con_arr_tensor_fft_mask_noise  #  tensor 1,1,256,256

            y = val_xdata.cuda() # 1 1 256 256
            gt = image_test.float().cuda() # 1 1 256 256
            with torch.no_grad():
                y_input = y  # 归一化之后的y 1 1 256 256
                zf = torch.fft.ifft2(y_input).cuda() * 256  # 1 1 256 256
                zf_input = abs(zf)  # 1 1 256 256 实数
                #####################
                zf_np = zf_input.squeeze(0).squeeze(0).detach().cpu().numpy()
                zf_np = zf_np * 255
                # cv2.imwrite(r'D:\code\xwy\CSMRI_Car_15\1\{}_rec.png'.format(filename), zf_np)
                # plt.imshow(zf_np, 'gray')
                # plt.show()
                #####################
                mask_val = mask_val.cuda()  # 1 1 256 256
# xu add
                Ax = con_arr_tensor_fft_mask.cuda()
                Ax_input = torch.fft.ifft2(Ax).cuda() * 256
                Ax_input = abs(Ax_input)

                x_hat, W1 = model(zf_input, y_input, mask_val, Ax_input, val=True)  ########重建图像
                # x_hat = torch.clamp(x_hat, 0., 1.)
            x_rec = x_hat  # 1 1 256 256
            #---------------------------------#
            rec_play =x_rec.squeeze(1).squeeze(0)
            rec_play = rec_play.detach().cpu().numpy() * 255
            cv2.imwrite(r'C:\Users\xu\Desktop\cartesian\result_15\{}_rec.png'.format(filename), rec_play)
            savemat(os.path.join(r'C:\Users\xu\Desktop\cartesian\mat_15', f'knee_radial_{filename}_{num_radial}.mat'),
                    {'rec_im': rec_play})
            # plt.imshow(rec_play, 'gray')
            # plt.show()
            #---------------------------------#
            psnr_val0, ssim_val0 = batch_PSNR_ssim(x_rec, gt, 1.)
            print("PSNR_val: {}, SSIM_val: {}".format(psnr_val0, ssim_val0))
            Pcount_psnr.append(psnr_val0)
            Pcount_ssim.append(ssim_val0)
            #-------------erro-pic------------#
            # gtnp = gt.squeeze(1).squeeze(0) # 256 256
            # gtnp = gtnp.detach().cpu().numpy()*255
            # erro = abs(rec_play - gtnp)
            # f = plt.figure()
            # ax = f.subplots()
            # im = ax.imshow(erro, cmap="rainbow")
            # plt.colorbar(mappable=im)
            # plt.savefig(r'F:\lz\DUconcsmriresultsabs\{}_erro.png'.format(filename))
            # plt.show()
            #---------------------------------#
        psnr_val = np.mean(Pcount_psnr)
        ssim_val = np.mean(Pcount_ssim)
        print("PSNR_val_mean: {}, SSIM_val_mean: {}".format(psnr_val, ssim_val))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConCSMRI")
    ##############常用参数############
    parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='15')
    parser.add_argument("--sigma", type=int, default=80)
    parser.add_argument("--depths", type=int, dest="depths", help="The depth of the network", default=8)
    parser.add_argument("--gamma", type=float, default=1.02)
    parser.add_argument("--beta", type=float, default=0.001)
    opt = parser.parse_args()  # 使用parse_args()解析添加的参数
    main(opt)
