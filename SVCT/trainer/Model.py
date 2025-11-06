import matplotlib.pyplot as plt
import torch.optim
from DoubleTightFrames.MyFilterLearningUtilize import *
from function import *

import torch





class CTnet(nn.Module):
    def __init__(self, channels=121):
        super(CTnet, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.channels)

        self.conv3 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(self.channels)

        self.conv5 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(self.channels)
        self.relu3 = nn.ReLU()
        self.conv6 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(self.channels)

        # self.sigmod = nn.Sigmoid()

    def forward(self, x):
        X = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        X1 = F.relu(X + out)
        out1 = self.conv3(X1)
        out1 = self.bn3(out1)
        out1 = self.relu2(out1)
        out1 = self.conv4(out1)
        out1 = self.bn4(out1)
        X2 = F.relu(X1 + out1)
        out2 = self.conv5(X2)
        out2 = self.bn5(out2)
        out2 = self.relu3(out2)
        out2 = self.conv6(out2)
        out2 = self.bn6(out2)
        X3 = F.relu(X2 + out2)

        return X3

class CTnet_nobn(nn.Module):
    def __init__(self, channels=121):
        super(CTnet_nobn, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.channels)

        self.conv3 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(self.channels)

        self.conv5 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(self.channels)
        self.relu3 = nn.ReLU()
        self.conv6 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn6 = nn.BatchNorm2d(self.channels)

        self.conv7 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv8 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)

        self.conv9 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.conv10 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.conv12 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)



    def forward(self, x):
        X = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        X1 = F.relu(X + out)

        out1 = self.conv3(X1)
        # out1 = self.bn3(out1)
        out1 = self.relu2(out1)
        out1 = self.conv4(out1)
        # out1 = self.bn4(out1)
        X2 = F.relu(X1 + out1)

        out2 = self.conv5(X2)
        # out2 = self.bn5(out2)
        out2 = self.relu3(out2)
        out2 = self.conv6(out2)
        # out2 = self.bn6(out2)
        X3 = F.relu(X2 + out2)

        out3 = self.conv7(X3)
        out3 = self.relu4(out3)
        out3 = self.conv8(out3)
        X4 = F.relu(X3 + out3)

        out4 = self.conv9(X4)
        out4 = self.relu5(out4)
        out4 = self.conv10(out4)
        X5 = F.relu(X4 + out4)

        out5 = self.conv11(X5)
        out5 = self.relu6(out5)
        out5 = self.conv12(out5)
        X6 = F.relu(X5 + out5)

        return X6

class CTnet_nobn_norelu(nn.Module):
    def __init__(self, channels=121):
        super(CTnet_nobn_norelu, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.channels)

        self.conv3 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn4 = nn.BatchNorm2d(self.channels)

        self.conv5 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(self.channels)
        self.relu3 = nn.ReLU()
        self.conv6 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn6 = nn.BatchNorm2d(self.channels)

        # self.sigmod = nn.Sigmoid()

    def forward(self, x):
        X = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        X1 = X + out
        out1 = self.conv3(X1)
        # out1 = self.bn3(out1)
        out1 = self.relu2(out1)
        out1 = self.conv4(out1)
        # out1 = self.bn4(out1)
        X2 = X1 + out1
        out2 = self.conv5(X2)
        # out2 = self.bn5(out2)
        out2 = self.relu3(out2)
        out2 = self.conv6(out2)
        # out2 = self.bn6(out2)
        X3 = X2 + out2

        return X3

class CTnet_nobn2(nn.Module):
    def __init__(self, channels=121):
        super(CTnet_nobn2, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(self.channels)

        self.conv3 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # # self.bn3 = nn.BatchNorm2d(self.channels)
        self.relu2 = nn.ReLU()
        self.conv4 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # # self.bn4 = nn.BatchNorm2d(self.channels)
        #
        # self.conv5 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # # self.bn5 = nn.BatchNorm2d(self.channels)
        # self.relu3 = nn.ReLU()
        # self.conv6 = nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1)
        # # self.bn6 = nn.BatchNorm2d(self.channels)
        #
        # # self.sigmod = nn.Sigmoid()

    def forward(self, x):
        X = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        X1 = F.relu(X + out)
        out1 = self.conv3(X1)
        # # out1 = self.bn3(out1)
        out1 = self.relu2(out1)
        out1 = self.conv4(out1)
        # # out1 = self.bn4(out1)
        X2 = F.relu(X1 + out1)
        # out2 = self.conv5(X2)
        # # out2 = self.bn5(out2)
        # out2 = self.relu3(out2)
        # out2 = self.conv6(out2)
        # # out2 = self.bn6(out2)
        # X3 = F.relu(X2 + out2)

        return X2

class network(nn.Module):
    def __init__(self, layers_num=5, sigma=80, beta=0.5, gamma=1.0006):
        super(network, self).__init__()
        self.depth = layers_num
        self.sigma = sigma
        self.beta = beta
        self.gamma = gamma
        self.taunet = CTnet_nobn().cuda()
        ####################
        layers = []
        for _ in range(self.depth):
            single_layer = CSMRI_Onelayer()
            layers.append(single_layer)  # 封装
        self.net = nn.Sequential(*layers)  # 容器

    def forward(self, zf, y, mask):
        x = Variable(zf.cuda())  # 1 1 256 256
        y_gpu = Variable(y.cuda())  # 1 1 256 256 复数
        # y_gpu = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])  # 1 1 256 256
        mask_gpu = Variable(mask.cuda())
        # sigma_map
        Inputsigma = self.sigma / 255
        sigma_hat = Variable(torch.cuda.FloatTensor([Inputsigma]))
        N, C, H, W = x.size()
        sigma_map = sigma_hat.view(N, 1, 1, 1).repeat(1, C, H, W)
        #
        for i in range(self.depth):
            beta_input = self.beta * (self.gamma ** i)
            x, W1, W2_loss, sigma = self.net[i](x, y_gpu, mask_gpu, sigma_map, beta_input)
            # 更新参数sigma_map
            tau = self.taunet(x)
            sigma_map = sigma * tau
        return x, W1, W2_loss  # x 1 1 256 256

class CSMRI_Onelayer(nn.Module):
    def __init__(self):
        super(CSMRI_Onelayer, self).__init__()
        # self.denoiser = DoubleTFlearningCNet_SE().cuda()
        self.denoiser = DoubleTFlearningCNet().cuda()
        # self.denoiser.apply(weights_init_kaiming)
        self.denoiser.setdecfilter()

    def forward(self, x, y_gpu, mask_gpu, sigma_map, beta):  # x 1 1 256 256 (取模值之后） y 1 1 256 256 复数  mask 1 1 256 256
        # input的x的维度为1 1 256 256，output的x的维度应与之一致
        #####去噪步骤
        x_hat, W1, W2loss = self.denoiser(x, sigma_map)
        x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256
        #####反演步骤
        F_x_hat = torch.fft.fft2(x_hat) / 256
        rec = (y_gpu * mask_gpu + beta * (F_x_hat * mask_gpu) + (1 + beta) * F_x_hat * (1 - mask_gpu)) / (1 + beta)
        rec = abs(torch.fft.ifft2(rec) * 256)
        sigma = sigma_map
        return rec, W1, W2loss, sigma

class Projnet(nn.Module):
    def __init__(self):
        super(Projnet, self).__init__()
        self.channels = 49
        self.T = 3
        self.layer = self.make_resblock(self.T)
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        S = input
        for i in range(self.T):
            S = S + self.layer[i](S)
        return S

class FEB_lamb(nn.Module):
    def __init__(self):
        super(FEB_lamb, self).__init__()
        self.channels = 1
        self.T = 3
        self.layer = self.make_resblock(self.T)
        self.apply_1 = torch.nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1)
        self.apply_2 = torch.nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1)
        self.apply_3 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.Tanh = nn.Tanh()

    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        feature_1 = self.apply_1(input)
        for i in range(self.T):
            feature_2 = self.layer[i](feature_1)
        feature_3 = feature_1 + feature_2
        feature_4 = self.apply_2(feature_3)
        feature_5 = self.apply_3(feature_4)
        out = self.Tanh(feature_5)
        out = abs(out)
        return out

class network_sct(nn.Module):
    def __init__(self,args):
        super(network_sct, self).__init__()
        self.depth = args.depth
        self.sigma = args.sigma
        self.beta = args.beta
        self.gamma = args.gamma
        self.taunet = CTnet_nobn().cuda()

        ####################
        # self.tau = 0.5
        # self.tau = 0.09
        self.tau = 0.09
        # self.lamb = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.lamb = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma_backtracking = 0.1
        self.eta_backtracking = 0.9


        self.eta1 =0.01
        self.eta2 =0.1

        self.eta3 = 0.01
        self.eta4 = 0.1

        self.eta5 = 0.01
        self.eta6 = 0.1

        self.eta7 = 0.01
        self.eta8 = 0.1

        self.eta9 = 0.01
        self.eta10 = 0.1

        self.eta11 = 0.01
        self.eta12 = 0.1


        self.eta1 = torch.tensor(self.eta1, dtype=torch.float32)
        self.eta2 = torch.tensor(self.eta2, dtype=torch.float32)

        self.eta3 = torch.tensor(self.eta3, dtype=torch.float32)
        self.eta4 = torch.tensor(self.eta4, dtype=torch.float32)

        self.eta5 = torch.tensor(self.eta5, dtype=torch.float32)
        self.eta6 = torch.tensor(self.eta6, dtype=torch.float32)

        self.eta7 = torch.tensor(self.eta7, dtype=torch.float32)
        self.eta8 = torch.tensor(self.eta8, dtype=torch.float32)

        self.eta9 = torch.tensor(self.eta9, dtype=torch.float32)
        self.eta10 = torch.tensor(self.eta10, dtype=torch.float32)

        self.eta11 = torch.tensor(self.eta11, dtype=torch.float32)
        self.eta12 = torch.tensor(self.eta12, dtype=torch.float32)

        self.kernel_size = 7
        self.num_filters = self.kernel_size ** 2
        self.conv_w = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0, bias=False)  ### 卷积不做padding
        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充# weight initial
        self.TNet_CNN_6 = TNet_CNN_1_2()
        self.TNet_CNN = TNet_CNN_3_6()
        self.TNet_SCT = TNet_SCT()
        self.apply_A = torch.nn.Conv2d(1, 31, kernel_size=3, stride=1, padding=1)
        self.apply_B = torch.nn.Conv2d(1, 31, kernel_size=3, stride=1, padding=1)
        self.apply_C = torch.nn.Conv2d(81, 49, kernel_size=3, stride=1, padding=1)
        self.atten = CALayer(81)
        self.resnet = Projnet()
        # self.lamblayer = FEB_lamb()

        # self.trainable_eta_0 = nn.Parameter(self.eta0,requires_grad=True)
        self.trainable_eta_1 = nn.Parameter(self.eta1,requires_grad=True)

        self.trainable_eta_2 = nn.Parameter(self.eta2,requires_grad=True)

        self.trainable_eta_3 = nn.Parameter(self.eta3, requires_grad=True)

        self.trainable_eta_4 = nn.Parameter(self.eta4, requires_grad=True)

        self.trainable_eta_5 = nn.Parameter(self.eta5, requires_grad=True)

        self.trainable_eta_6 = nn.Parameter(self.eta6, requires_grad=True)

        self.trainable_eta_7 = nn.Parameter(self.eta7, requires_grad=True)

        self.trainable_eta_8 = nn.Parameter(self.eta8, requires_grad=True)

        self.trainable_eta_9 = nn.Parameter(self.eta9, requires_grad=True)

        self.trainable_eta_10 = nn.Parameter(self.eta10, requires_grad=True)

        self.trainable_eta_11 = nn.Parameter(self.eta11, requires_grad=True)

        self.trainable_eta_12 = nn.Parameter(self.eta12, requires_grad=True)

    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w.weight.shape).permute(0, 1, 3, 2).contiguous()

    def forward(self, zf, y, mask, Ax_input,radon,iradon, val):
        x = Variable(zf.cuda())  # 1 1 256 256
        y_gpu = Variable(y.cuda())  # 1 1 256 256 复数
        # y_gpu = torch.complex(y[:, :, :, :, 0], y[:, :, :, :, 1])  # 1 1 256 256
        mask_gpu = Variable(mask.cuda())

        img_tensor = x

        F_back = 1
        backtracking_check = True
        i = 0
        while i < self.depth:
            F_old = F_back
            x_old = x
            x_0 = x
            if i == 0:
                # lamb = self.lamblayer(x_0)
                beta_input = self.beta * (self.gamma ** i)
                # 去噪步骤
                Wx = self.conv_w(self.rpad(x_0))  # conv_w：紧标架
                # epsilon = self.TNet_CNN_6(Wx)  # 计算epsilon
                epsilon = self.resnet(Wx)  # 计算epsilon
                epsilon = torch.clamp(epsilon, 1e-5, 10)
                z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
                weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
                Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
                temp = Wt
                weight = self.conv_w.weight
                x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
                for j in range(3):
                    rec = x_old - 2 * self.trainable_eta_1 * iradon(
                        mask_gpu @ radon(x_old)) + 2 * self.trainable_eta_1 * iradon(
                        mask_gpu @ y_gpu) - 2 * self.trainable_eta_2 * (x_old - x_hat)
                    # 更新 x_old
                    x_old = rec
                # x_hat=temp

                # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (1 - mask_gpu)) / (
                #             1 + beta_input)
                # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
                # out = x
                # x_1 = x
                # max=torch.max(x_hat)
                # min=torch.min(x_hat)

                # x_hat=(x_hat-torch.min(x_hat))/(torch.max(x_hat)-torch.min((x_hat)))



                # L_x_hat = (radon(x_hat))
                #
                # L_x_hat_max = torch.max(L_x_hat)
                # L_x_hat_min = torch.min(L_x_hat)




                # L_x_hat = L_x_hat / torch.max(L_x_hat)


                # rec = (self.eta* mask_gpu @ y_gpu + beta_input * (mask_gpu@L_x_hat) + (self.eta+ beta_input) * (1 - mask_gpu)@ L_x_hat  ) / (
                #             self.eta + beta_input)
                # y_gpu=(y_gpu-torch.min(y_gpu))/(torch.max(y_gpu)-torch.min(y_gpu))
                # L_x_hat=(L_x_hat-torch.min(L_x_hat))/(torch.max(L_x_hat)-torch.min(L_x_hat))

                # L1 =  torch.zeros((1, 1,360, 800), device='cuda')
                # L1[:,:, 0:360:6, :] = 1
                # b1 = L1
                # # L_sampleing = (mask_gpu @ y_gpu + self.trainable_eta_0 *mask_gpu @(L_x_hat)) / (1 + self.trainable_eta_0)
                # L_sampleing = (mask_gpu @ y_gpu + self.trainable_eta_0 * mask_gpu @ (L_x_hat)) / (
                #             1 + self.trainable_eta_0)
                # L_x_hat_max = torch.max(mask_gpu @(L_x_hat))
                # L_x_hat_min = torch.min(mask_gpu @(L_x_hat))
                #
                # identity_matrix = torch.eye(360).cuda()
                #
                # # 调整维度为 (1, 1, 360, 360)
                # I = identity_matrix.unsqueeze(0).unsqueeze(0)
                #
                # # L_unsampleing = (I - mask_gpu) @ L_x_hat
                # L_sampleing[1:361:6, :]= L_x_hat[1:361:6, :]
                # L_sampleing[2:361:6, :] = L_x_hat[2:361:6, :]
                # L_sampleing[3:361:6, :] = L_x_hat[3:361:6, :]
                # L_sampleing[4:361:6, :] = L_x_hat[4:361:6, :]
                # L_sampleing[5:361:6, :] = L_x_hat[5:361:6, :]
                #
                # # L_unsampleing = L_x_hat * b2
                # # L_unsampleing=L_unsampleing.detach().cpu().numpy()
                # # L=L_sampleing + L_unsampleing
                # L = L_sampleing
                # # L=(L-torch.min(L))/(torch.max(L)-torch.min(L))
                # x = iradon(L)

                # plt.imshow(x.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()




                # rec = (mask_gpu @ y_gpu + self.trainable_eta_0 * (mask_gpu @ L_x_hat) + (1+ self.trainable_eta_0) * (
                #             1 - mask_gpu) @ L_x_hat) / (
                #                       1 + self.trainable_eta_0)
                # for i in range(mask_gpu.shape[0]):
                #     if np.all(mask_gpu[i, :] == 0):  # 当前行全为 0，表示未采样
                #         rec = Lz[i, i]  # 保留原值
                #     elif Omega[i, i] == 1:  # 当前行的对角线元素为 1，表示采样
                #         Lx_next[i, i] = (epsilon_0 + beta * Lz[i, i]) / (epsilon + beta)

                # x = iradon(rec)

                # x = (x - torch.min(x)) / (
                #         torch.max(x) - torch.min(x))
                # x = torch.clamp(x, 0, 1)
                out = rec
                x_1 = rec


            if i == 1:
                # lamb = self.lamblayer(x_1)
                beta_input = self.beta * (self.gamma ** i)

                Wx = self.conv_w(self.rpad(x_1))  # conv_w：紧标架
                # epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
                epsilon_1 = self.resnet(Wx)
                x_0_in = self.apply_A(x_0)
                x_1_in = self.apply_B(x_1)
                epsilon_2 = self.TNet_SCT(x_1, x_0_in, x_1_in)
                epsilon = torch.cat((epsilon_1, epsilon_2), 1)
                epsilon = self.atten(epsilon)
                epsilon = self.apply_C(epsilon)
                epsilon = torch.clamp(epsilon, 1e-5, 10)
                z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
                weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
                Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
                temp = Wt
                weight = self.conv_w.weight

                x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
                for j in range(3):
                    rec=x_old-2*self.trainable_eta_3*iradon(
                    mask_gpu @radon(x_old))+2*self.trainable_eta_3*iradon(mask_gpu@y_gpu)-2*self.trainable_eta_4*(x_old-x_hat)
                    x_old = rec

               # plt.imshow(x_hat.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()
                # x_hat = temp

# xu
#                 x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256  # 去噪图像

                #####反演步骤
                # F_x_hat = torch.fft.fft2(x_hat) / 256
                # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (
                #             1 - mask_gpu)) / (
                #               1 + beta_input)
                # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
                # x_hat = (x_hat - torch.min(x_hat)) / (torch.max(x_hat) - torch.min((x_hat)))
                # L_x_hat = radon(x_hat)
                #
                # L_x_hat_max_1 = torch.max(L_x_hat)
                # L_x_hat_min_1 = torch.min(L_x_hat)
                #
                #
                #
                #
                # # L_x_hat = L_x_hat / torch.max(L_x_hat)
                #
                # # rec = (self.eta* mask_gpu @ y_gpu + beta_input * (mask_gpu@L_x_hat) + (self.eta+ beta_input) * (1 - mask_gpu)@ L_x_hat  ) / (
                # #             self.eta + beta_input)
                # L1 = torch.zeros((1, 1, 360, 800), device='cuda')
                # L1[:,:, 0:360:6, :] = 1
                # b1 = L1
                #
                # # y_gpu = (y_gpu - torch.min(y_gpu)) / (torch.max(y_gpu) - torch.min(y_gpu))
                # # L_x_hat = (L_x_hat - torch.min(L_x_hat)) / (torch.max(L_x_hat) - torch.min(L_x_hat))
                #
                #
                #
                # L_sampleing = (mask_gpu @ y_gpu + self.trainable_eta_1 *mask_gpu @ (L_x_hat)) / (1 + self.trainable_eta_1)
                #
                # identity_matrix = torch.eye(360).cuda()
                #
                # # 调整维度为 (1, 1, 360, 360)
                # I = identity_matrix.unsqueeze(0).unsqueeze(0)
                #
                # L_sampleing[1:361:6, :] = L_x_hat[1:361:6, :]
                # L_sampleing[2:361:6, :] = L_x_hat[2:361:6, :]
                # L_sampleing[3:361:6, :] = L_x_hat[3:361:6, :]
                # L_sampleing[4:361:6, :] = L_x_hat[4:361:6, :]
                # L_sampleing[5:361:6, :] = L_x_hat[5:361:6, :]
                #
                # # L_unsampleing = L_x_hat * b2
                # # L_unsampleing=L_unsampleing.detach().cpu().numpy()
                # # L=L_sampleing + L_unsampleing
                # L = L_sampleing
                # # L=(L-torch.min(L))/(torch.max(L)-torch.min(L))
                # x = iradon(L)
                # # x = (x - torch.min(x)) / (
                # #         torch.max(x) - torch.min(x))
                # # x = torch.clamp(x, 0, 1)

                out = rec
                x_2 = rec

            if i == 2:
                # lamb = self.lamblayer(x_2)
                beta_input = self.beta * (self.gamma ** i)

                Wx = self.conv_w(self.rpad(x_2))  # conv_w：紧标架
                # epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
                epsilon_1 = self.resnet(Wx)
                x_1_in = self.apply_A(x_1)
                x_2_in = self.apply_B(x_2)
                epsilon_2 = self.TNet_SCT(x_2, x_1_in, x_2_in)
                epsilon = torch.cat((epsilon_1, epsilon_2), 1)
                epsilon = self.atten(epsilon)
                epsilon = self.apply_C(epsilon)
                epsilon = torch.clamp(epsilon, 1e-5, 10)
                z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
                weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
                Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
                temp = Wt
                weight = self.conv_w.weight

                x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
                for j in range(3):
                    rec = x_old - 2 * self.trainable_eta_5* iradon(
                    mask_gpu @radon(x_old)) + 2 * self.trainable_eta_5 * iradon(mask_gpu @y_gpu) - 2 * self.trainable_eta_6 * (
                                  x_old - x_hat)
                    x_old = rec

                # x_hat = temp
                # plt.imshow(x_hat.squeeze().cpu().detach().numpy(), cmap='gray')
                # plt.show()
                # xu
                #                 x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256  # 去噪图像

                #####反演步骤
                # F_x_hat = torch.fft.fft2(x_hat) / 256
                # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (
                #         1 - mask_gpu)) / (
                #               1 + beta_input)
                # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
                # L_x_hat = radon(x_hat/255)/4.0*255
                # x_hat = (x_hat - torch.min(x_hat)) / (torch.max(x_hat) - torch.min((x_hat)))
                # L_x_hat = radon(x_hat)
                # # L_x_hat =  L_x_hat / torch.max(L_x_hat)
                #
                # # L_x_hat_max_2 = torch.max(L_x_hat)
                # # L_x_hat_min_2 = torch.min(L_x_hat)
                # L_x_hat_max = torch.max(mask_gpu @ (L_x_hat))
                # L_x_hat_min = torch.min(mask_gpu @ (L_x_hat))
                #
                # # rec = (self.eta* mask_gpu @ y_gpu + beta_input * (mask_gpu@L_x_hat) + (self.eta+ beta_input) * (1 - mask_gpu)@ L_x_hat  ) / (
                # #             self.eta + beta_input)
                # L1 = torch.zeros((1, 1, 360, 800), device='cuda')
                # L1[:,:,0:360:6, :] = 1
                # b1 = L1
                #
                # # y_gpu = (y_gpu - torch.min(y_gpu)) / (torch.max(y_gpu) - torch.min(y_gpu))
                # # L_x_hat = (L_x_hat - torch.min(L_x_hat)) / (torch.max(L_x_hat) - torch.min(L_x_hat))
                #
                # L_sampleing = ((mask_gpu @ y_gpu + self.trainable_eta_2 *mask_gpu @ (L_x_hat)) / (1 + self.trainable_eta_2))
                #
                # # L2 = torch.ones((1, 1, 360, 800), device='cuda')
                # #
                # # L2[:,:, 0:360:6, :] = 0
                # # b2 = L2
                # identity_matrix = torch.eye(360).cuda()
                #
                # # 调整维度为 (1, 1, 360, 360)
                # I = identity_matrix.unsqueeze(0).unsqueeze(0)
                #
                # L_sampleing[1:361:6, :] = L_x_hat[1:361:6, :]
                # L_sampleing[2:361:6, :] = L_x_hat[2:361:6, :]
                # L_sampleing[3:361:6, :] = L_x_hat[3:361:6, :]
                # L_sampleing[4:361:6, :] = L_x_hat[4:361:6, :]
                # L_sampleing[5:361:6, :] = L_x_hat[5:361:6, :]
                #
                # # L_unsampleing = L_x_hat * b2
                # # L_unsampleing=L_unsampleing.detach().cpu().numpy()
                # # L=L_sampleing + L_unsampleing
                # L = L_sampleing
                # # L=(L-torch.min(L))/(torch.max(L)-torch.min(L))
                # x = iradon(L)
                # x = (x - torch.min(x)) / (
                #         torch.max(x) - torch.min(x))
                # x = torch.clamp(x, 0, 1)
                out = rec
                x_3 = rec

            if i == 3:
                # lamb = self.lamblayer(x_3)
                beta_input = self.beta * (self.gamma ** i)

                Wx = self.conv_w(self.rpad(x_3))  # conv_w：紧标架
                # epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
                epsilon_1 = self.resnet(Wx)
                x_2_in = self.apply_A(x_2)
                x_3_in = self.apply_B(x_3)
                epsilon_2 = self.TNet_SCT(x_3, x_2_in, x_3_in)
                epsilon = torch.cat((epsilon_1, epsilon_2), 1)
                epsilon = self.atten(epsilon)
                epsilon = self.apply_C(epsilon)
                epsilon = torch.clamp(epsilon, 1e-5, 10)
                z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
                weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
                Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
                temp = Wt
                weight = self.conv_w.weight

                x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
                # xu
                #                 x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256  # 去噪图像

                #####反演步骤
                # F_x_hat = torch.fft.fft2(x_hat) / 256
                # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (
                #         1 - mask_gpu)) / (
                #               1 + beta_input)
                # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
                for j in range(3):
                    rec = x_old - 2 * self.trainable_eta_7 * iradon(
                    mask_gpu @ radon(x_old)) + 2 * self.trainable_eta_7 * iradon(
                    mask_gpu @ y_gpu) - 2 * self.trainable_eta_8 * (
                              x_old - x_hat)
                    x_old = rec

                out = rec
                x_4 = rec
            # #
            if i == 4:
                # lamb = self.lamblayer(x_4)
                beta_input = self.beta * (self.gamma ** i)

                Wx = self.conv_w(self.rpad(x_4))  # conv_w：紧标架
                # epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
                epsilon_1 = self.resnet(Wx)
                x_3_in = self.apply_A(x_3)
                x_4_in = self.apply_B(x_4)
                epsilon_2 = self.TNet_SCT(x_4, x_3_in, x_4_in)
                epsilon = torch.cat((epsilon_1, epsilon_2), 1)
                epsilon = self.atten(epsilon)
                epsilon = self.apply_C(epsilon)
                epsilon = torch.clamp(epsilon, 1e-5, 10)
                z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
                weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
                Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
                temp = Wt
                weight = self.conv_w.weight

                x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
                # xu
                #                 x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256  # 去噪图像

                #####反演步骤
                # F_x_hat = torch.fft.fft2(x_hat) / 256
                # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (
                #         1 - mask_gpu)) / (
                #               1 + beta_input)
                # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
                for j in range(3):
                    rec = x_old - 2 * self.trainable_eta_9 * iradon(
                    mask_gpu @ radon(x_old)) + 2 * self.trainable_eta_9 * iradon(
                    mask_gpu @ y_gpu) - 2 * self.trainable_eta_10 * (
                              x_old - x_hat)
                    x_old = rec

                out = rec
                x_5 = rec
            #
            # if i == 5:
            #     # lamb = self.lamblayer(x_5)
            #     beta_input = self.beta * (self.gamma ** i)
            #
            #     Wx = self.conv_w(self.rpad(x_5))  # conv_w：紧标架
            #     # epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
            #     epsilon_1 = self.resnet(Wx)
            #     x_4_in = self.apply_A(x_4)
            #     x_5_in = self.apply_B(x_5)
            #     epsilon_2 = self.TNet_SCT(x_5, x_4_in, x_5_in)
            #     epsilon = torch.cat((epsilon_1, epsilon_2), 1)
            #     epsilon = self.atten(epsilon)
            #     epsilon = self.apply_C(epsilon)
            #     epsilon = torch.clamp(epsilon, 1e-5, 10)
            #     z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
            #     weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
            #     Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
            #     temp = Wt
            #     weight = self.conv_w.weight
            #
            #     x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
            #     # xu
            #     #                 x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256  # 去噪图像
            #
            #     #####反演步骤
            #     # F_x_hat = torch.fft.fft2(x_hat) / 256
            #     # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (
            #     #         1 - mask_gpu)) / (
            #     #               1 + beta_input)
            #     # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
            #     rec = x_old - 2 * self.trainable_eta_1 * iradon(
            #         mask_gpu @ radon(x_old)) + 2 * self.trainable_eta_5 * iradon(
            #         mask_gpu @ y_gpu) - 2 * self.trainable_eta_6 * (
            #                   x_old - x_hat)
            #     out = rec
            #     x_6 = x
            #
            # if i == 6:
            #     # lamb = self.lamblayer(x_6)
            #     beta_input = self.beta * (self.gamma ** i)
            #
            #     Wx = self.conv_w(self.rpad(x_6))  # conv_w：紧标架
            #     # epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
            #     epsilon_1 = self.resnet(Wx)
            #     x_5_in = self.apply_A(x_5)
            #     x_6_in = self.apply_B(x_6)
            #     epsilon_2 = self.TNet_SCT(x_6, x_5_in, x_6_in)
            #     epsilon = torch.cat((epsilon_1, epsilon_2), 1)
            #     epsilon = self.atten(epsilon)
            #     epsilon = self.apply_C(epsilon)
            #     epsilon = torch.clamp(epsilon, 1e-5, 10)
            #     z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
            #     weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
            #     Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
            #     temp = Wt
            #     weight = self.conv_w.weight
            #
            #     x_hat = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
            #     # xu
            #     #                 x_hat = torch.clamp(x_hat, 0., 1.)  # 1 1 256 256  # 去噪图像
            #
            #     #####反演步骤
            #     # F_x_hat = torch.fft.fft2(x_hat) / 256
            #     # rec = (y_gpu * mask_gpu + beta_input * (F_x_hat * mask_gpu) + (1 + beta_input) * F_x_hat * (
            #     #         1 - mask_gpu)) / (
            #     #               1 + beta_input)
            #     # x = abs(torch.fft.ifft2(rec) * 256)  # 重建图像
            #     L_x_hat = radon(x_hat)
            #
            #     rec = (self.eta * mask_gpu @ y_gpu + beta_input * (mask_gpu @ L_x_hat) + (self.eta + beta_input) * (
            #             1 - mask_gpu) @ L_x_hat) / (
            #                   self.eta + beta_input)
            #     x = abs(iradon(rec))
            #     out = x
            #     x_7 = x

            f = 0.5 * torch.norm(img_tensor - Ax_input, p=2) ** 2
            r = 0.5 * torch.sum((x_old - temp).reshape((x_old.shape[0], -1)) ** 2)
            F_back = f + self.lamb * r

            # Backtracking
            if i > 1:
                diff_x = (torch.norm(x - x_old, p=2) ** 2)
                diff_F = F_old - F_back
                if diff_F < (self.gamma_backtracking / self.tau) * diff_x and self.tau > 0.1 and val==False:
                    self.tau = self.eta_backtracking * self.tau
                    backtracking_check = False
                    print('backtracking : tau =', self.tau, 'diff_F=', diff_F, 'diff_x=', diff_x)
                else:
                    backtracking_check = True

            if backtracking_check:  # if the backtracking condition is satisfied
                i =i+ 1  # next iteration

            else:  # if the backtracking condition is not satisfied
                x = x_old
                F_back = F_old


        return out, weight





# class SCTNet(nn.Module):
#     def __init__(self, n_iter=8):
#         super(SCTNet, self).__init__()
#         self.n = n_iter
#
#         self.tau = 0.5
#         self.lamb = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#         self.gamma_backtracking = 0.1
#         self.eta_backtracking = 0.9
#
#         self.kernel_size = 11
#         self.num_filters = self.kernel_size ** 2
#         self.conv_w = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0, bias=False)  ### 卷积不做padding
#         self.soft_threshold = SoftThreshold(self.num_filters, 1)
#         self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充# weight initial
#         self.TNet_CNN_6 = TNet_CNN_1_2()
#         self.TNet_CNN = TNet_CNN_3_6()
#         self.TNet_SCT = TNet_SCT()
#         self.apply_A = torch.nn.Conv2d(1, 31_ori, kernel_size=3, stride=1, padding=1)
#         self.apply_B = torch.nn.Conv2d(1, 31_ori, kernel_size=3, stride=1, padding=1)
#         self.apply_C = torch.nn.Conv2d(153, 121, kernel_size=3, stride=1, padding=1)
#
#         self.atten = CALayer(153)
#
#     def setdecfilter(self):
#         # mat_dict = scipy.io.loadmat('globalTF9.mat')
#         mat_dict = scipy.io.loadmat('globalTF11.mat')
#         tframedict = mat_dict['learnt_dict'].transpose()
#         self.conv_w.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w.weight.shape).permute(0, 1, 3, 2).contiguous()
#
#     def forward(self, x, k, sf, sigma, val):
#         img_tensor = x
#         w, h = x.shape[-2:]
#         FB = p2o(k, (w * sf, h * sf))
#         FBC = cconj(FB, inplace=False)
#         F2B = r2c(cabs2(FB))
#         STy = upsample(x, sf=sf)
#         FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
#         x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
#
#         ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))
#
#         F_back = 1
#         backtracking_check = True
#         i = 0
#         while i < self.n:
#             F_old = F_back
#             x_old = x
#             x_0 = x
#             if i == 0:
#                 Wx = self.conv_w(self.rpad(x_0))  # conv_w：紧标架
#                 epsilon = self.TNet_CNN_6(Wx)  # 计算epsilon
#                 # epsilon = self.atten(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_1 = x
#
#             if i == 1:
#                 Wx = self.conv_w(self.rpad(x_1))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_0_in = self.apply_A(x_0)
#                 x_1_in = self.apply_B(x_1)
#                 epsilon_2 = self.TNet_SCT(x_1, x_0_in, x_1_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_2 = x
#
#             if i == 2:
#                 Wx = self.conv_w(self.rpad(x_2))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_1_in = self.apply_A(x_1)
#                 x_2_in = self.apply_B(x_2)
#                 epsilon_2 = self.TNet_SCT(x_2, x_1_in, x_2_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_3 = x
#
#             if i == 3:
#                 Wx = self.conv_w(self.rpad(x_3))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_2_in = self.apply_A(x_2)
#                 x_3_in = self.apply_B(x_3)
#                 epsilon_2 = self.TNet_SCT(x_3, x_2_in, x_3_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_4 = x
#
#             if i == 4:
#                 Wx = self.conv_w(self.rpad(x_4))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_3_in = self.apply_A(x_3)
#                 x_4_in = self.apply_B(x_4)
#                 epsilon_2 = self.TNet_SCT(x_4, x_3_in, x_4_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_5 = x
#
#             if i == 5:
#                 Wx = self.conv_w(self.rpad(x_5))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_4_in = self.apply_A(x_4)
#                 x_5_in = self.apply_B(x_5)
#                 epsilon_2 = self.TNet_SCT(x_5, x_4_in, x_5_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_6 = x
#
#             if i == 6:
#                 Wx = self.conv_w(self.rpad(x_6))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_5_in = self.apply_A(x_5)
#                 x_6_in = self.apply_B(x_6)
#                 epsilon_2 = self.TNet_SCT(x_6, x_5_in, x_6_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_7 = x
#
#             if i == 7:
#                 Wx = self.conv_w(self.rpad(x_7))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_6_in = self.apply_A(x_6)
#                 x_7_in = self.apply_B(x_7)
#                 epsilon_2 = self.TNet_SCT(x_7, x_6_in, x_7_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_8 = x
#
#             if i == 8:
#                 Wx = self.conv_w(self.rpad(x_8))  # conv_w：紧标架
#                 epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#                 x_7_in = self.apply_A(x_7)
#                 x_8_in = self.apply_B(x_8)
#                 epsilon_2 = self.TNet_SCT(x_8, x_7_in, x_8_in)
#                 epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#                 epsilon = self.atten(epsilon)
#                 epsilon = self.apply_C(epsilon)
#                 epsilon = torch.clamp(epsilon, 1e-5, 10)
#                 z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#                 weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#                 Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#                 temp = Wt
#                 weight = self.conv_w.weight
#                 z = (1 - self.tau * self.lamb) * x_old + self.tau * self.lamb * temp
#                 x = self.d(z, FB, FBC, F2B, FBFy, ab[:, i:i + 1, ...], sf)
#                 y = z  # output image is the output of the denoising step
#                 x_9 = x
#
#
#             x_out = torch.zeros(1, 1, img_tensor.shape[2], img_tensor.shape[3]).cuda()
#             for j in range(x_old.shape[0]):
#                 x_in = x_old[j, ...].unsqueeze(0)
#                 Ax_1 = utils_sr.G(x_in, k[j, 0, ...], sf)  # Calculation A*x with A the linear degradation operator
#                 x_out = torch.cat((x_out, Ax_1), dim=0)
#             num = x_out.shape[0]
#             Ax = x_out[1:num, ...]
#             f = 0.5 * torch.norm(img_tensor - Ax, p=2) ** 2
#             r = 0.5 * torch.sum((x_old - temp).reshape((x_old.shape[0], -1)) ** 2)
#             F_back = f + self.lamb * r
#
#             # Backtracking
#             if i > 1:
#                 diff_x = (torch.norm(x - x_old, p=2) ** 2)
#                 diff_F = F_old - F_back
#                 if diff_F < (self.gamma_backtracking / self.tau) * diff_x and self.tau > 0.1 and val==False:
#                     self.tau = self.eta_backtracking * self.tau
#                     backtracking_check = False
#                     print('backtracking : tau =', self.tau, 'diff_F=', diff_F, 'diff_x=', diff_x)
#                 else:
#                     backtracking_check = True
#
#             if backtracking_check:  # if the backtracking condition is satisfied
#                 i += 1  # next iteration
#
#             else:  # if the backtracking condition is not satisfied
#                 x = x_old
#                 F_back = F_old
#
#         return y, weight