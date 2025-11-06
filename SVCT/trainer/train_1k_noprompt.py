from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch
import time
import numpy as np
from skimage.metrics import structural_similarity as skssim
from skimage.metrics import peak_signal_noise_ratio as skpsnr
from utils import ssim_loss
from torch.utils.tensorboard import SummaryWriter
#from model import MetaInvNet_H
from torch import nn
import matplotlib.pyplot as plt
from utils.pytorch_msssim import MS_SSIM
import random
import torch.backends.cudnn as cudnn
from head_HM import init_logger
import torch.nn.functional as F
from segment_anything import sam_model_registry
# device = args.druevice
# medsam_model = sam_model_registry["vit_b"](checkpoint=r'D:\hlj\ampnet\work_dir\MedSAM\medsam_vit_b.pth')
# medsam_model = medsam_model.cuda()
# medsam_model.eval()
from Model import *
class Trainer():
    def __init__(self, args, model, tr_dset, vl_dset,test_dset):
        self.args = args
        self.model = model
        self.epoch = args.epoch
        self.bat_size = args.tr_batch
        self.valbat_size = args.vl_batch
        self.tr_dset = tr_dset
        self.vl_dset = vl_dset
        self.training_params = {}
        self.best_psnr = 0
        self.best_psnr_epoch = 0
        self.test_dset=test_dset
        self.proj, self.back_proj, self.fbp, self.op_norm = self.test_dset.radon_transform_120(num_view=120)



    def tr(self):
        # ——————————————————————随机种子—————————————————————————
        if self.args.manualSeed is None:
            self.args.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.args.manualSeed)
        random.seed(self.args.manualSeed)
        torch.manual_seed(self.args.manualSeed)
        cudnn.benchmark = True
        # —————————————————————
        writer = SummaryWriter("./logs_train")
        logger = init_logger(self.args)

        self.model = self.model.cuda()
        # self.model.setdecfilter()
        num = self.print_network(self.model)
        logger.info("\tTotal number of parameters: {}".format(num))

        self.optimizer = optim.Adam(self.model.parameters(), betas=(0.5, 0.999), lr=self.args.lr )
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=0.89)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestone, gamma=0.5)

        criterion = my_two_TF_sum_squared_error().cuda()

        start = 0
        if self.args.resume ==True:
            start = self.resume_tr()
        print('************train dataset length:{}************'.format(len(self.tr_dset)))
        DLoader = DataLoader(dataset=self.tr_dset, num_workers=0, drop_last=True, batch_size=self.bat_size,shuffle=True)#shuffle=True
        DLoaderval = DataLoader(dataset=self.vl_dset, num_workers=0, drop_last=True, batch_size=self.valbat_size,shuffle=True)
        # —————————————————————

        for epoch in range(start,self.epoch):
            t_start = time.time()
            # current_lr = self.args.lr
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"当前学习率：{current_lr}")
            loss_per_epoch_tr = 0
            psnr_per_epoch_tr = 0
            ssim_per_epoch_tr = 0
            for n_count, data_batch in enumerate(DLoader):
                bat_num = len(DLoader)
                if n_count % 50 == 0:
                    print('\n   ********** This is the training stage *************')
                batch_img, batch_u0, batch_sino,sino_60, imgSAM, angle,img_1024,sgt = [x.cuda() for x in data_batch]

                xt = self.iradon(sgt)
                # psnr_iter = self.aver_psnr(batch_img, xt)
                # plt.imshow(batch_img.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()
                #
                # plt.imshow(xt.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()


                A = np.zeros((self.bat_size, 1, 360, 360))
                A = torch.from_numpy(A).cuda()
                b = torch.eye(360)
                b = np.asarray(b)
                b[1:361:3, :] = 0
                b[2:362:3, :] = 0

                # b[4:364:6, :] = 0
                # b[5:365:6, :] = 0
                b = torch.tensor(b)
                A[:, :, :, :] = b
                A = A.type(torch.FloatTensor).cuda()

                A1 = np.zeros((self.valbat_size, 1, 360, 360))
                A1 = torch.from_numpy(A1).cuda()
                A1[:, :, :, :] = b
                A1 = A1.type(torch.FloatTensor).cuda()
                # A=A.cpu().numpy()
                # print(A)

                y = batch_sino.cuda()  # y (1,1,256,256) 复数
                y_input = y  # 归一化之后的y
                # gt = image_train.float().cuda()  # gt (1,1,256,256)
                zf=self.fbp(batch_sino)
                # plt.imshow(batch_sino.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()

                # zf = torch.fft.ifft2(y_input).cuda() * 256  # 1 1 256 256
                zf_input = zf  # 1 1 256 256

                zf_max=torch.max(zf)
                zf_min=torch.min(zf)

                # xu add
                Ax = sino_60.cuda()
                # Ax_input = torch.fft.ifft2(Ax).cuda() * 256
                Ax_input = self.fbp(Ax)
                Ax_input = Ax_input

                # plt.imshow(Ax_input.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()

                max=torch.max(batch_sino)
                min=torch.min(batch_sino)

                # with torch.no_grad():
                #     image_embedding = medsam_model.image_encoder(img_1024)
                    # image_embedding = image_embedding.cpu()

                    # 取出前5个通道,每个通道画为一张图
                    # f, axarr = plt.subplots(1, 2)
                    # for i in range(2):
                    #     axarr[i].imshow(image_embedding[0, i, :, :])
                    #     axarr[i].set_title('Channel {}'.format(i))
                    #
                    # plt.show()
    #                 image_embedding=  F.interpolate(
    #     image_embedding,
    #     size=(512, 512),
    #     mode="bilinear",
    #     align_corners=False,
    # )  # (1, 1, gt.shape)
                # image_embedding = image_embedding[:, :32, :, :]
                self.model.train()
                self.optimizer.zero_grad()
                # tau = max(1 - (epoch - 1) / 100, 0.4)

                # weiying,batch_x_out = self.model(batch_u0,imgSAM,tau)
                x_hat,W1 = self.model(zf_input,y_input,A,Ax_input,self.radon, self.iradon, val = False)
                # batch_x_out=batch_x_out[0]
                # batch_x_out=batch_x_out.unsqueeze(1)

                # batch_img = F.interpolate(batch_img, size=(416, 416), mode='bilinear', align_corners=False)
                # loss = self.loss_dd(batch_x_out, batch_img)+0.2*self.loss_dd(weiying,batch_img-batch_u0)
                # image_train = Variable(image_train.cuda())
                # xu
                # loss = criterion(x_hat, image_train, W1, opt.lamuda1) / (zf.size()[0] * 2) + opt.lamuda2 * W2_loss / (
                #             zf.size()[0] ** 2)
                loss = criterion(x_hat, batch_img, W1, 0.0001) / (zf.size()[0] * 2)
                loss.backward()
                self.optimizer.step()



                loss_iter = loss.item()
                psnr_iter = self.aver_psnr(x_hat, batch_img)
                ssim_iter = self.aver_ssim(x_hat, batch_img)
                loss_per_epoch_tr += loss_iter
                psnr_per_epoch_tr += psnr_iter.item()
                ssim_per_epoch_tr += ssim_iter.item()

            # if n_count % 5 == 0:
                template = '[***] Epoch {} of {}, Batch {} of {}, loss={:5.2f}, psnr={:4.2f}, ssim={:5.4f}, lr={:.2e}'
                print(template.format(epoch + 1, self.epoch, n_count, bat_num, loss_iter, psnr_iter, ssim_iter, current_lr))

            loss_per_epoch_tr /= (n_count + 1)
            psnr_per_epoch_tr /= (n_count + 1)
            ssim_per_epoch_tr /= (n_count + 1)
            print('Train: Loss={:+.2e} PSNR={:4.2f} SSIM={:5.4f}'.format(loss_per_epoch_tr, psnr_per_epoch_tr,
                                                                         ssim_per_epoch_tr))
            # ———————————————————— val stage————————————————————————————————————
            print('\n   ********** This is the validation stage *************')
            print('************val dataset length:{}************'.format(len(self.vl_dset)))
            rmse_per_epoch_vl = 0
            psnr_per_epoch_vl = 0
            ssim_per_epoch_vl = 0
            self.model.eval()
            for n_count, data_batch in enumerate(DLoaderval):
                val_img, val_u0, val_sino,sino_gt_60, imgSAM, angle,img_1024_tensor = [x.cuda() for x in data_batch]
                # with torch.no_grad():
                #     image_embedding = medsam_model.image_encoder(img_1024)
                #     image_embedding = F.interpolate(
                #         image_embedding,
                #         size=(512, 512),
                #         mode="bilinear",
                #         align_corners=False,
                #     )  # (1, 1, gt.shape)
                # image_embedding = image_embedding[:, :32, :, :]
                y = val_sino.cuda()  # y (1,1,256,256) 复数
                y_input = y  # 归一化之后的y
                # gt = image_train.float().cuda()  # gt (1,1,256,256)
                zf = self.iradon(val_sino)
                # zf = torch.fft.ifft2(y_input).cuda() * 256  # 1 1 256 256
                zf_input_val = zf  # 1 1 256 256

                # xu add
                Ax_val = sino_gt_60.cuda()
                # Ax_input = torch.fft.ifft2(Ax).cuda() * 256
                Ax_val = self.iradon(Ax_val)
                # Ax_input = Ax_input

                with torch.no_grad():

                    # val_x_db = self.model(val_sino,val_u0,self.radon, self.iradon,imgSAM,tau)
                    val_x_db,_ = self.model(zf_input_val,y_input,A, Ax_val,self.radon, self.iradon, val = False)
                psnr_iter = self.aver_psnr(val_x_db, val_img)
                ssim_iter = self.aver_ssim(val_x_db, val_img)
                rmse_iter = torch.sqrt(torch.mean((val_x_db - val_img) ** 2))
                psnr_per_epoch_vl += psnr_iter.item()
                ssim_per_epoch_vl += ssim_iter.item()
                rmse_per_epoch_vl += rmse_iter.item()
                log_str = '[***] Epoch {} of {}, val:{:0>3d}, psnr={:4.2f}, ssim={:5.4f}, rmse={:5.4f} lr={:.2e} '
                print(log_str.format(epoch + 1, self.epoch, n_count + 1, psnr_iter, ssim_iter, rmse_iter,current_lr))

            psnr_per_epoch_vl /= (n_count + 1)
            ssim_per_epoch_vl /= (n_count + 1)
            rmse_per_epoch_vl /= (n_count + 1)
            print('val PSNR mean:{}'.format(psnr_per_epoch_vl))
            print('val SSIM mean:{}'.format(ssim_per_epoch_vl))
            print('val RMSE mean:{}'.format(rmse_per_epoch_vl))

            t_end = time.time()
            time_ = t_end - t_start
            print(' One Epoch consumes time= %2.2f' % (time_))

         # ————————————————————————————— save best model————————————————————————
            if (epoch + 1) % 1 == 0 or epoch == self.epoch - 1:
                #self.save_ckp(epoch)

                if psnr_per_epoch_vl > self.best_psnr:
                    self.best_psnr = psnr_per_epoch_vl
                    self.training_params['best_psnr'] = self.best_psnr
                    self.training_params['best_psnr_epoch'] = epoch + 1
                    self.best_psnr_epoch = epoch + 1
                    model_filename_best = self.args.model_dir + 'ddnet_best.pth'
                    self.save_ckp(model_filename_best)

                model_filename = self.args.model_dir + 'epoch%d.pth' % (epoch + 1)
                self.save_ckp(model_filename)

                logger.info(
                    "\tval: current_epoch:{}  psnr_val:{:.4f} ssim_val:{:.4f} rmse_val:{:.4f} time:{:.2f}  best_psnr:{:.4f}  best_psnr_epoch:{}"
                    .format(epoch + 1, psnr_per_epoch_vl, ssim_per_epoch_vl, rmse_per_epoch_vl, time_, self.best_psnr, self.best_psnr_epoch))
                print('-' * 100)

                print("best_psnr:{:.4f} best_psnr_epoch:{}".format(self.best_psnr, self.best_psnr_epoch))

                print('\n' + '--->' * 10 + 'Save CKP now！！!')

                writer.add_scalar('tr2 PSNR epoch', psnr_per_epoch_tr, epoch + 1)
                writer.add_scalar('tr2 SSIM epoch', ssim_per_epoch_tr, epoch + 1)
                writer.add_scalar('tr2 loss epoch', loss_per_epoch_tr, epoch + 1)
                # writer.add_scalar('tr RMSE_epoch', rmse_per_epoch_tr, epoch + 1)

                writer.add_scalar('val5 RMSE_epoch', rmse_per_epoch_vl, epoch + 1)
                writer.add_scalar('val5 PSNR epoch', psnr_per_epoch_vl, epoch + 1)
                writer.add_scalar('val5 SSIM epoch', ssim_per_epoch_vl, epoch + 1)
                # writer.add_scalar('val loss epoch', loss_per_epoch_vl, epoch + 1)
                writer.add_scalar('Learning rate4', current_lr, epoch + 1)

            writer.close()
            self.scheduler.step()
        print('Reach the maximal epoch! Finish training')

    def loss_dd(self, db, sp):
        loss1 = nn.MSELoss(reduction='sum')  # sum
        # for ii in range(0, layer):
        #     #print("sp.shape , db[ii].shape",sp.shape , db[ii].shape)   #torch.Size([1, 1, 512, 512]) torch.Size([1, 512, 512])
        #     loss =loss+ l2loss_mean(sp, db[ii].unsqueeze(0)) * 1.1**ii

        loss_1 = loss1(db, sp) / sp.size(0)
        # loss_1 = F.mse_loss(db, sp)
        # loss_2 = MS_SSIM(sp,db)  # (N,)ms_ssim
        losser2 = MS_SSIM(data_range=1.0).cuda()
        loss_2 = losser2(sp, db).mean()
        loss = loss_1 + 0.1 * loss_2
        # loss = loss_1

        # #loss =0
        # loss1 = l2loss_mean(sp, l1)*1.1**1
        # loss2 = l2loss_mean(sp, l2) * 1.1 ** 2
        # loss3 = l2loss_mean(sp, l3) * 1.1 ** 3
        # loss = l2loss_mean(sp, db) * 1.1 ** 4+loss3+loss1+loss2
        # # loss=loss+l2loss_mean(self.radon(sp),self.radon(db))
        # # # for jj in range(sp.shape[1]):
        # # #     img1 = sp[jj, ...]
        # # # #     img2 = db[jj, ...]
        # img1 = sp.squeeze(0)
        # img2 = db.squeeze(0)
        #
        # img3 = l1.squeeze(0)
        # img4 = l2.squeeze(0)
        # img5 = l3.squeeze(0)
        #
        #
        # ssim_value1 = 1 - ssim_loss(img1, img2, size=11, sigma=1.5)
        # ssim_value2 = 1 - ssim_loss(img1, img3, size=11, sigma=1.5)
        # ssim_value3 = 1 - ssim_loss(img1, img4, size=11, sigma=1.5)
        # ssim_value4 = 1 - ssim_loss(img1, img5, size=11, sigma=1.5)
        #
        # loss = loss + ssim_value1 *100+ssim_value2 *100+ssim_value3 *100+ssim_value4 *100
        return loss

    def resume_tr(self):
        ckp = torch.load(self.args.resume_ckp_dir)
        #print(ckp['model'])
        self.model.load_state_dict(ckp['model'], False)
        self.optimizer.load_state_dict(ckp['optimizer'])
        self.scheduler.load_state_dict(ckp['scheduler'])
        self.training_params = ckp['training_params']
        print("self.training_params", self.training_params)
        self.best_psnr = self.training_params['best_psnr']
        self.best_psnr_epoch = self.training_params['best_psnr_epoch']
        return int(self.args.resume_ckp_resume)

    def print_network(self,net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print('Total number of parameters: %d' % num_params)
        return num_params
    #打印wangluo参数个数

    def save_ckp(self, filename):
        #filename = self.model_dir + 'epoch%d.pth' % (epoch+1)
        #print(self.training_params)
        state = {'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'training_params': self.training_params
                 }
        torch.save(state,filename)

    def psnr(self,img1, img2):
        if isinstance(img1, torch.Tensor):
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        psnr = skpsnr(img1.cpu().detach().numpy(), img2.cpu().detach().numpy(), data_range=1.0)
        #tensor转numpy计算
        return psnr

    def aver_psnr(self, img1, img2):
        PSNR = 0
        assert img1.shape == img2.shape
        for i in range(img1.shape[0]):
            for j in range(img1.shape[1]):
                PSNR += self.psnr(img1[i, j:j + 1, ...], img2[i, j:j + 1, ...])
        return PSNR / (img1.shape[0] * img1.shape[1])

    def aver_ssim(self, img1, img2):
        '''used in the training'''
        # from skimage.measure import compare_ssim as ski_ssim
        SSIM = 0
        img1 = img1.detach().cpu().numpy().astype(np.float64)
        img2 = img2.detach().cpu().numpy().astype(np.float64)
        for i in range(len(img1)):
            for j in range(img1.shape[1]):
                SSIM += skssim(img1[i, j, ...], img2[i, j, ...], gaussian_weights=True, win_size=11, data_range=1.0, sigma=1.5)
        #a = len(img1)
        return SSIM / (len(img1) * img1.shape[1])
    def radon(self,img):
        if len(img.shape)==4:
            img=img.squeeze(1)
        return self.proj(img).unsqueeze(1)
    # def iradon(self,sino):
    #     if len(sino.shape)==4:
    #         sino=sino.squeeze(1)
    #     return self.back_proj(sino).unsqueeze(1)

    def iradon(self, sino):
        if len(sino.shape) == 4:
            sino = sino.squeeze(1)
        return self.fbp(sino).unsqueeze(1)

