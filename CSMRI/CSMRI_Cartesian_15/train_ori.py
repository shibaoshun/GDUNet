import os
import argparse
# import h5py
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import torch.nn as nn
from Model import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 设备号 默认为0
torch.manual_seed(0)
def main(opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    L = np.array([55])  # 采样率
    sigma = 0.1  # 噪声水平
    #----------采样率设置----------
    for index in range(len(L)):
        num_radial = L[index]
        Pcount_psnr = []
        Pcount_ssim = []
        Pcount_ssim_mean = []
        Pcount_psnr_mean = []
        logger = init_logger(opt)
        sys.stdout = Logger(sys.stdout)
        #----------模型，优化器，loss设置等----------
# XU
        model = network(layers_num=opt.depths, sigma=opt.sigma, beta=opt.beta, gamma=opt.gamma).cuda()
        # model = network_sct(layers_num=opt.depths, sigma=opt.sigma, beta=opt.beta, gamma=opt.gamma).cuda()
        # model.setdecfilter()

        num_params = print_network(model)
        print('Total number of parameters: %d' % num_params)
        # device_ids = [0]
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestone, gamma=opt.lr_decay)
        criterion = my_two_TF_sum_squared_error().cuda()
        #----------断线重连----------
        config_dict = {
        'name': opt.model_name,
        'depth': opt.depths,
        'sigma': opt.sigma,
        'gamma': opt.gamma,
        'beta': opt.beta,
        'num_epochs': opt.num_epochs,
        'lr': opt.lr,
        'lr_decay': opt.lr_decay,
        'milestone': opt.milestone,
        'params': num_params
        }
        print(config_dict)
        with open(f'{opt.out_dir}/{opt.model_name}.config', 'w') as txt_file: txt_file.write(str(config_dict))
        if opt.resume_training:
            resumef = os.path.join(opt.out_dir, 'ckpt.pth')
            if os.path.isfile(resumef):
                checkpoint = torch.load(resumef)
                print("> Resuming previous training")
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                new_epoch = opt.num_epochs
                new_milestone = opt.milestone
                opt = checkpoint['args']
                training_params = checkpoint['training_params']
                start_epoch = training_params['start_epoch']
                best_psnr = training_params['best_psnr']
                opt.num_epochs = new_epoch
                opt.milestone = new_milestone
                print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
                print("=> loaded parameters :")
                print("==> checkpoint['optimizer']['param_groups']")
                print("\t{}".format(checkpoint['optimizer']['param_groups']))
                print("==> checkpoint['training_params']")
                for k in checkpoint['training_params']:
                    print("\t{}, {}".format(k, checkpoint['training_params'][k]))
                argpri = vars(checkpoint['args'])
                print("==> checkpoint['args']")
                for k in argpri:
                    print("\t{}, {}".format(k, argpri[k]))
                opt.resume_training = False
            else:
                raise Exception("Couldn't resume training with checkpoint {}".format(resumef))
        else:
            training_params = {}
            start_epoch = 0
            best_psnr = 0
            training_params['step'] = 0
        #----------训练----------
        for epoch in range(start_epoch, opt.num_epochs):
            epoch_loss_train = 0
            epoch_samples_train = 0
            psnr_train_average = 0
            ssim_train_average = 0
            countt = 0
            time_start = time.time()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(current_lr)
            #-----数据加载-----
            dataset_train = lkxDataset(train=True)
            dataset_val = lkxDataset(train=False)
            loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)
            loader_val = DataLoader(dataset=dataset_val, num_workers=0, batch_size=opt.batchSize, shuffle=True)
            for iii, image_train in enumerate(loader_train, 0):  # tensor 1,1,256,256
                countt = countt + 1
                image_train = image_train.float() / 255  # tensor 1 1 256 256   0-1
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                # 创建采样矩阵
                train_gt_use_for_gen_mask_noise = image_train.squeeze(1).numpy()   #####1,256,256
                [m, u, v] = np.shape(train_gt_use_for_gen_mask_noise)
                mask = np.zeros([m, u, v])
                for i in range(0, m):
                    arr = train_gt_use_for_gen_mask_noise[i, :, :]
                    mask0 = gen_mask(arr, num_radial)
                    mask[i, :, :] = mask0  # 1,256,256
                mask = torch.tensor(mask).unsqueeze(-1).float().cuda()####tensor 1,256,256,1
                mask_input = mask.permute(0, 3, 1, 2)  # 1 1 256 256  tensor   1在四周
                # 创建噪声矩阵
                noise = sigma/255 * torch.randn(256, 256) + sigma /255 * torch.randn(256, 256) * (1.j)
                noise_input = noise.unsqueeze(0).unsqueeze(1).cuda()  # 1 1 256 256
                #
                con_arr_torch = torch.tensor(train_gt_use_for_gen_mask_noise).unsqueeze(-1).float()  # tensor 1 256 256 1
                con_arr = con_arr_torch.permute(0, 3, 1, 2)  # 1 1 256 256
                con_arr = torch.fft.fft2(con_arr).cuda() / 256  # 1 1 256 256   /n
                con_arr_tensor_fft_mask = con_arr * mask_input  # 1 1 256 256
                con_arr_tensor_fft_mask_noise = con_arr_tensor_fft_mask + noise_input  # tensor 1 1 256 256 复数
                train_xdata = con_arr_tensor_fft_mask_noise  # tensor 1,1,256,256

                y = train_xdata.cuda()  # y (1,1,256,256) 复数
                y_input = y   # 归一化之后的y
                gt = image_train.float().cuda()    # gt (1,1,256,256)
                zf = torch.fft.ifft2(y_input).cuda() * 256  # 1 1 256 256
                zf_input = abs(zf)  # 1 1 256 256

# xu add
                Ax = con_arr_tensor_fft_mask.cuda()
                Ax_input = torch.fft.ifft2(Ax).cuda() * 256
                Ax_input = abs(Ax_input)

                ######
                # zf_play =abs(zf_input).squeeze(1).squeeze(0)
                # zf_play = zf_play.detach().cpu().numpy()
                # plt.imshow(zf_play, 'gray')
                # plt.show()
                #####
# xu
                x_hat, W1, W2_loss = model(zf_input, y_input, mask_input)  ########重建图像 1 1 256 256
                image_train = Variable(image_train.cuda())
                # x_hat, W1 = model(zf_input, y_input, mask_input, Ax_input, val = False)  ########重建图像 1 1 256 256
                # image_train = Variable(image_train.cuda())
# xu
                loss = criterion(x_hat, image_train, W1, opt.lamuda1) / (zf.size()[0] * 2) + opt.lamuda2 * W2_loss / (
                            zf.size()[0] ** 2)
                # loss = criterion(x_hat, image_train, W1, opt.lamuda1) / (zf.size()[0] * 2)
                loss.backward()
                optimizer.step()

                epoch_loss_train += loss.data.cpu().numpy()
                epoch_samples_train += len(y)

                x_rec = x_hat  # 1 1 256 256
                psnr_train, ssim_train = batch_PSNR_ssim(x_rec, gt, 1.)
                psnr_train_average += psnr_train
                ssim_train_average += ssim_train

                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f, SSIM_train: %.4f, Current_lr: %.8f" % (epoch + 1, iii + 1, len(loader_train), loss.item(), psnr_train, ssim_train, current_lr))
                training_params['step'] += 1
                sys.stdout.flush()

            epoch_loss_train /= epoch_samples_train
            train_epoch_loss = epoch_loss_train
            psnr_train_average /= countt
            ssim_train_average /= countt
            print("train_epoch_loss = {}, psnr_train_average = {}, ssim_train_average = {}".format(train_epoch_loss, psnr_train_average, ssim_train_average))

            scheduler.step()
            epoch_loss_val = 0
            epoch_samples_val = 0
            count = 0
            for iiii, image_val in enumerate(loader_val, 0):
                count = count + 1
                image_val = image_val.float() / 255
                model.eval()
                val_gt_use_for_gen_mask_noise = image_val.squeeze(1).numpy()  # 1 256 256
                [m, u, v] = np.shape(val_gt_use_for_gen_mask_noise)
                # mask
                mask = np.zeros([m, u, v])
                for i in range(0, m):
                    arr = val_gt_use_for_gen_mask_noise[i, :, :]
                    mask0 = gen_mask(arr, num_radial)
                    mask[i, :, :] = mask0  # 1,256,256
                mask = torch.tensor(mask).unsqueeze(-1).float()  # tensor 1,256,256,1
                mask_val = mask.permute(0,3,1,2).cuda()  # 1 1 256 256
                # noise
                noise = sigma/255 * torch.randn(256, 256) + sigma /255 * torch.randn(256, 256) * (1.j)
                noise_val = noise.unsqueeze(0).unsqueeze(1).cuda()  # 1 1 256 256
                #
                con_arr_torch = torch.tensor(val_gt_use_for_gen_mask_noise).unsqueeze(-1).float() # 1 256 256 1
                con_arr = con_arr_torch.permute(0, 3, 1, 2)  # 1 1 256 256
                con_arr = torch.fft.fft2(con_arr).cuda() / 256  # 1 1 256 256
                con_arr_tensor_fft_mask = con_arr * mask_val
                con_arr_tensor_fft_mask_noise = con_arr_tensor_fft_mask + noise_val  # tensor 1 1 256 256
                val_xdata = con_arr_tensor_fft_mask_noise  # tensor 1,1,256,256



                y = val_xdata.cuda()  # 1 1 256 256
                gt = image_val.float().cuda()  # 1 1 256 256
                with torch.no_grad():
                    y_input = y  # 归一化之后的y 1 1 256 256
                    zf = torch.fft.ifft2(y_input).cuda() * 256  # 1 1 256 256
                    zf_input = abs(zf)  # 1 1 256 256 实数
                    mask_val = mask_val.cuda()  # 1 1 256 256
# xu add
                    Ax = con_arr_tensor_fft_mask.cuda()
                    Ax_input = torch.fft.ifft2(Ax).cuda() * 256
                    Ax_input = abs(Ax_input)
# xu
                    x_hat, W1, W2_loss = model(zf_input, y_input, mask_val)  ########重建图像
                    # x_hat, W1 = model(zf_input, y_input, mask_val, Ax_input, val = True)  ########重建图像
                    image_val = Variable(image_val.cuda())  # 1 1 256 256
# xu
                    loss = criterion(x_hat, image_val, W1, opt.lamuda1) / (zf.size()[0] * 2) + opt.lamuda2 * W2_loss / (
                                zf.size()[0] ** 2)
                    # loss = criterion(x_hat, image_val, W1, opt.lamuda1) / (zf.size()[0] * 2)
                    epoch_loss_val += loss.data.cpu().numpy()
                epoch_samples_val += len(y)

                x_rec = x_hat  # 1 1 256 256
                psnr_val0, ssim_val0 = batch_PSNR_ssim(x_rec, gt, 1.)
                print("\n[epoch %d][%d/%d]  PSNR_val: %.4f,  SSIM_val: %.4f" % (epoch + 1, count, len(loader_val), psnr_val0, ssim_val0))
                Pcount_psnr.append(psnr_val0)
                Pcount_ssim.append(ssim_val0)

            psnr_val = np.mean(Pcount_psnr)
            ssim_val = np.mean(Pcount_ssim)
            time_end = time.time()
            cur_time = time_end - time_start
            print("\n[epoch %d] PSNR_val: %.4f, SSIM_val: %.4f, Time： %.2f" % (epoch + 1, psnr_val, ssim_val, cur_time))
            sys.stdout.flush()
            training_params['start_epoch'] = epoch + 1
            Pcount_psnr_mean.append(psnr_val)
            Pcount_ssim_mean.append(ssim_val)
            # save
            if psnr_val > best_psnr:
                best_psnr = psnr_val
                training_params['best_psnr'] = best_psnr
                torch.save(model.state_dict(), f'{opt.out_dir}/{opt.model_name}.model')  # 保存best_model

            if (epoch+1) % 10 == 0 and (epoch+1) > 0:  # 每10代输出结果
                torch.save(model.state_dict(), f'{opt.out_dir}/{epoch+1}.model')

            save_dict = {'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'scheduler': scheduler.state_dict(),
                         'training_params': training_params,
                         'args': opt}
            torch.save(save_dict, os.path.join(opt.out_dir, 'ckpt.pth'))  # 保存current model

            epoch_loss_val /= epoch_samples_val
            val_epoch_loss = epoch_loss_val
            print("val_epoch_loss=", val_epoch_loss)
            logger.info("\tcurrent_epoch:{}  PSNR_val:{:.4f}  SSIM_val:{:.4f}  best_psnr:{:.4f}  TRAINLOSS:{:.4f}  VALLOSS:{:.4f}  currentlr:{:.4f}".format(epoch + 1, psnr_val, ssim_val, best_psnr, train_epoch_loss, val_epoch_loss, current_lr))
            sys.stdout.flush()

        # plt.plot(opt.num_epochs, Pcount_psnr_mean, label="PSNR VS. Ieration")
        # plt.xlabel("Iteration")
        # plt.ylabel("PSNR")
        # plt.title("{}".format(opt.model_name) )
        # plt.savefig("ConCSMRI_DU\example.png")
        # plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ConCSMRI")
    parser.add_argument("--resume_training", "--r", default=False, help="resume training from a previous checkpoint")
    # parser.add_argument("--resume_training", "--r", default=True, help="resume training from a previous checkpoint")
    ##############常用参数############
    parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results dir path", default='31_ori')
    parser.add_argument("--sigma", type=int, default=80)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=1.02)
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate")
    parser.add_argument("--depths", type=int, dest="depths", help="The depth of the network", default=2)
    ################################
    parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default='knee4000_radial')
    parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=[30], help="When to decay learning rate; should be less than epochs")
    parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.1)
    parser.add_argument("--lamuda1", type=float, default=0.001)
    parser.add_argument("--lamuda2", type=float, default=0.001)
    opt = parser.parse_args()  # 使用parse_args()解析添加的参数
    main(opt)