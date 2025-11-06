from __future__ import division
import torch.nn.functional as F
from torch.nn import init
import torch
import scipy.io
import torchvision
from torch import nn
import sys
import numbers
from einops import rearrange
import numpy as np
sys.path.append("..")

def CtoT(data):
    data = torch.cat((data.real, data.imag), dim=1)
    return data

def TtoC(data):
    data = torch.complex(data[:, 0:121, :, :], data[:, 122:243, :, :])  # 1 1 256 256
    return data

class SEAttention(nn.Module):
    def __init__(self, channel=121, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNet(nn.Module):
    def __init__(self, num_filters):
        super(CNet, self).__init__()
        self.apply_A = torch.nn.Conv2d(122, num_filters, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) # T 244
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # conv + relu
        temp1 = self.relu(self.apply_A(x))
        y = temp1
        return y

def conv_block_conv_tong_cnet(in_channels, out_channels):  # 一个卷积层一个relu放一起
    blk = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True), nn.ReLU())
    return blk

class DenseBlock_conv_tong_cnet(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):  # 3 121 64
        super(DenseBlock_conv_tong_cnet, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels
            net.append(conv_block_conv_tong_cnet(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # 计算输出通道数

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)    # Y 64*i
            X = torch.cat((X, Y), dim=1)  # 在通道维上将输入和输出连结
        return X

def transition_block_kernel_size3(in_channels, out_channels):
    blk = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
    return blk

def resnetblock(x, out):
    out_resnet = x+out
    return out_resnet

class SoftThreshold(nn.Module):
    # perfect soft function
    def __init__(self, size, init_threshold=1e-1):   # 121 1
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x, threshold):
        mask1 = (x > threshold).float()
        mask2 = (x < -threshold).float()
        out = mask1.float() * (x - threshold)
        out += mask2.float() * (x + threshold)
        return out

def rot180(d):
    # # the filtersize must be a 奇数
    d = d.permute(1, 0, 2, 3)
    filtersize = d.shape[3]
    a = d.clone()
    for i in range(1, (filtersize+1)):
        a[:, :, (i-1), :] = d[:, :, (filtersize-i), :]
    c = a.clone()
    for i in range(1, (filtersize+1)):
        c[:, :, :, (i-1)] = a[:, :, :, (filtersize-i)]
    return c

# # FSAS # #
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)
        output = v * out
        output = self.project_out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FSAS(dim, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x

#----------------------------------- 9.18 -----------------------------------#
class DoubleTFlearningCNet_SE(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_SE, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 3  # # 卷积层  数
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # Cnet
        self.Constantnet1 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Constantnet2 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Attention
        self.SE1 = SEAttention(121, 16)
        self.SE2 = SEAttention(121, 16)
        self.conv1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  ## 122个通道
        Constantnet_denseblock10 = self.Constantnet1(input1)
        Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121
        Constantnet1 = self.SE1(resnet_out1)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121

        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  #### 122
        Constantnet_denseblock20 = self.Constantnet2(input2)
        Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)
        Constantnet2 = self.SE2(resnet_out2)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121
        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss

#----------------------------------- 10.6 -----------------------------------#
#两个Cnet使用同一个Transformer（共享）显存过大
class DoubleTFlearningCNet_Transformer(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_Transformer, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 3  # # 卷积层  数
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # RDN
        # self.Constantnet1 = CNet(self.num_filters).cuda()
        # self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        # self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        #
        # self.Constantnet2 = CNet(self.num_filters).cuda()
        # self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        # self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Transformer
        self.Transformerblock1 = TransformerBlock(dim=122).cuda()
        # self.Transformerblock2 = TransformerBlock(dim=122).cuda()
        # # Attention
        self.SE1 = SEAttention(122, 16)
        self.SE2 = SEAttention(122, 16)
        self.conv1 = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  ## 122个通道 (1 122 256 256)
        #
        # input_fft1 = torch.fft.fft2(input1).cuda() / 256 # (1 122 256 256)
        # Constantnet_denseblock10 = self.Constantnet1(input_fft1)
        # Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        # Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        # resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121 (1 121 256 256)
        # output1 = torch.fft.ifft2(resnet_out1).cuda() * 256

        FSAS1 = self.Transformerblock1(input1)  #(1 122 256 256)
        out1 = FSAS1
        # out1 = torch.cat((output1, FSAS1), dim=1)
        Constantnet1 = self.SE1(out1)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        # # # # #
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  #### 122
        #
        # input_fft2 = torch.fft.fft2(input2).cuda() / 256
        # Constantnet_denseblock20 = self.Constantnet2(input_fft2)
        # Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        # Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        # resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)
        # output2 = torch.fft.ifft2(resnet_out2).cuda() * 256

        FSAS2 = self.Transformerblock1(input2)
        out2 = FSAS2
        # out2 = torch.cat((output2, FSAS2), dim=1)
        Constantnet2 = self.SE2(out2)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121
        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss
#------------------------------------10.8-----------------------------------#
#两个Cnet使用2个Transformer（非共享）显存过大
class DoubleTFlearningCNet_Transformer_2(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_Transformer_2, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 3  # # 卷积层  数
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # RDN
        # self.Constantnet1 = CNet(self.num_filters).cuda()
        # self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        # self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        #
        # self.Constantnet2 = CNet(self.num_filters).cuda()
        # self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        # self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Transformer
        self.Transformerblock1 = TransformerBlock(dim=122).cuda()
        self.Transformerblock2 = TransformerBlock(dim=122).cuda()
        # # Attention
        self.SE1 = SEAttention(122, 16)
        self.SE2 = SEAttention(122, 16)
        self.conv1 = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  ## 122个通道 (1 122 256 256)
        #
        # input_fft1 = torch.fft.fft2(input1).cuda() / 256 # (1 122 256 256)
        # Constantnet_denseblock10 = self.Constantnet1(input_fft1)
        # Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        # Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        # resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121 (1 121 256 256)
        # output1 = torch.fft.ifft2(resnet_out1).cuda() * 256

        FSAS1 = self.Transformerblock1(input1)  #(1 122 256 256)
        out1 = FSAS1
        # out1 = torch.cat((output1, FSAS1), dim=1)
        Constantnet1 = self.SE1(out1)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        # # # # #
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  #### 122
        #
        # input_fft2 = torch.fft.fft2(input2).cuda() / 256
        # Constantnet_denseblock20 = self.Constantnet2(input_fft2)
        # Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        # Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        # resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)
        # output2 = torch.fft.ifft2(resnet_out2).cuda() * 256

        FSAS2 = self.Transformerblock2(input2)
        out2 = FSAS2
        # out2 = torch.cat((output2, FSAS2), dim=1)
        Constantnet2 = self.SE2(out2)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121
        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss
#完全共享1个Cnet
class DoubleTFlearningCNet_Transformer_3(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_Transformer_3, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 3  # # 卷积层  数
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # RDN
        # self.Constantnet1 = CNet(self.num_filters).cuda()
        # self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        # self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        # self.Constantnet2 = CNet(self.num_filters).cuda()
        # self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        # self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Transformer
        self.Transformerblock1 = TransformerBlock(dim=122).cuda()
        # self.Transformerblock2 = TransformerBlock(dim=122).cuda()
        # # Attention
        self.SE1 = SEAttention(122, 16)
        # self.SE2 = SEAttention(122, 16)
        self.conv1 = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # self.conv2 = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  ## 122个通道 (1 122 256 256)
        #
        # input_fft1 = torch.fft.fft2(input1).cuda() / 256 # (1 122 256 256)
        # Constantnet_denseblock10 = self.Constantnet1(input_fft1)
        # Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        # Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        # resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121 (1 121 256 256)
        # output1 = torch.fft.ifft2(resnet_out1).cuda() * 256

        FSAS1 = self.Transformerblock1(input1)  #(1 122 256 256)
        out1 = FSAS1
        # out1 = torch.cat((output1, FSAS1), dim=1)
        Constantnet1 = self.SE1(out1)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        # # # # #
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  #### 122
        #
        # input_fft2 = torch.fft.fft2(input2).cuda() / 256
        # Constantnet_denseblock20 = self.Constantnet2(input_fft2)
        # Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        # Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        # resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)
        # output2 = torch.fft.ifft2(resnet_out2).cuda() * 256

        FSAS2 = self.Transformerblock1(input2)
        out2 = FSAS2
        # out2 = torch.cat((output2, FSAS2), dim=1)
        Constantnet2 = self.SE1(out2)
        Constantnet2 = self.conv1(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121
        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss
#------------------------------------FFT_RDN----------------------------------#
class DoubleTFlearningCNet_FFTRDN(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_FFTRDN, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 3  # # 卷积层  数
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # Cnet
        self.Constantnet1 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Constantnet2 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Attention
        self.SE1 = SEAttention(121, 16)
        self.SE2 = SEAttention(121, 16)
        self.conv1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  ## 122个通道

        input1_fft = torch.fft.fft2(input1).cuda() / 256  # (1 122 256 256)
        real = input1_fft.real # 1 122 256 256
        imag = input1_fft.imag # 1 122 256 256

        Constantnet_denseblock10 = self.Constantnet1(real)
        Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121

        Constantnet_denseblock101 = self.Constantnet1(imag)
        Constantnet_denseblock111 = self.Constantnet_denseblock11(Constantnet_denseblock101)
        Constantnet_denseblock121 = self.Constantnet_denseblock12(Constantnet_denseblock111)
        resnet_out11 = resnetblock(Constantnet_denseblock101, Constantnet_denseblock121)  ## 121+121 = 121
        out1 = torch.complex(resnet_out1, resnet_out11)  # 1 121 256 256 复数
        out1_ifft = torch.fft.ifft2(out1).cuda() * 256 # 1 121 256 256 实数
        out1_ifft = abs(out1_ifft)

        Constantnet1 = self.SE1(out1_ifft)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        ##########################################
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  #### 122

        input2_fft = torch.fft.fft2(input2).cuda() / 256  # (1 122 256 256)
        real2 = input2_fft.real  # 1 122 256 256
        imag2 = input2_fft.imag  # 1 122 256 256

        Constantnet_denseblock20 = self.Constantnet2(real2)
        Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)

        Constantnet_denseblock201 = self.Constantnet2(imag2)
        Constantnet_denseblock211 = self.Constantnet_denseblock21(Constantnet_denseblock201)
        Constantnet_denseblock221 = self.Constantnet_denseblock22(Constantnet_denseblock211)
        resnet_out21 = resnetblock(Constantnet_denseblock201, Constantnet_denseblock221)
        out2 = torch.complex(resnet_out2, resnet_out21)  # 1 121 256 256 复数
        out2_ifft = torch.fft.ifft2(out2).cuda() * 256  # 1 121 256 256 实数
        out2_ifft = abs(out2_ifft)

        Constantnet2 = self.SE2(out2_ifft)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121

        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss

#----------------------------------------10.14--------------------------#
# Transformer 级联 RDN #
class DoubleTFlearningCNet_FFTRDN_Transformer(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_FFTRDN_Transformer, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 2  # # 卷积层  数 原 3
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # RDN
        self.Constantnet1 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Constantnet2 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Transformer
        self.Transformerblock = TransformerBlock(dim=122).cuda()
        self.conv = torch.nn.Conv2d(122, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Attention
        self.SE1 = SEAttention(242, 16)
        self.SE2 = SEAttention(242, 16)
        self.conv1 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  ## 122个通道
        # # FFT RDN FFT # #
        input1_fft = torch.fft.fft2(input1).cuda() / 256  # (1 122 256 256)
        real = input1_fft.real # 1 122 256 256
        imag = input1_fft.imag # 1 122 256 256

        Constantnet_denseblock10 = self.Constantnet1(real)
        Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121

        Constantnet_denseblock101 = self.Constantnet1(imag)
        Constantnet_denseblock111 = self.Constantnet_denseblock11(Constantnet_denseblock101)
        Constantnet_denseblock121 = self.Constantnet_denseblock12(Constantnet_denseblock111)
        resnet_out11 = resnetblock(Constantnet_denseblock101, Constantnet_denseblock121)  ## 121+121 = 121
        out1 = torch.complex(resnet_out1, resnet_out11)  # 1 121 256 256 复数
        out1_ifft = torch.fft.ifft2(out1).cuda() * 256 # 1 121 256 256 复数
        out1_ifft = abs(out1_ifft)
        # # Transformer # #
        FSAS1 = self.Transformerblock(input1) # 1 122 256 256
        FSAS1_1 = self.conv(FSAS1) # 1 121 256 256
        # print(FSAS1_1.shape)
        # # 级联
        Cnet1_out = torch.cat((out1_ifft, FSAS1_1), dim=1) #1 242 256 256
        # print(Cnet1_out.shape)
        Constantnet1 = self.SE1(Cnet1_out)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        ##########################################
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  #### 122

        input2_fft = torch.fft.fft2(input2).cuda() / 256  # (1 122 256 256)
        real2 = input2_fft.real  # 1 122 256 256
        imag2 = input2_fft.imag  # 1 122 256 256

        Constantnet_denseblock20 = self.Constantnet2(real2)
        Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)

        Constantnet_denseblock201 = self.Constantnet2(imag2)
        Constantnet_denseblock211 = self.Constantnet_denseblock21(Constantnet_denseblock201)
        Constantnet_denseblock221 = self.Constantnet_denseblock22(Constantnet_denseblock211)
        resnet_out21 = resnetblock(Constantnet_denseblock201, Constantnet_denseblock221)
        out2 = torch.complex(resnet_out2, resnet_out21)  # 1 121 256 256 复数
        out2_ifft = torch.fft.ifft2(out2).cuda() * 256  # 1 121 256 256 复数
        out2_ifft = abs(out2_ifft)
        # # Transformer # #
        FSAS2 = self.Transformerblock(input2)  # 1 122 256 256
        FSAS2_1 = self.conv(FSAS2)  # 1 121 256 256

        # # 级联
        Cnet2_out = torch.cat((out2_ifft, FSAS2_1), dim=1)

        Constantnet2 = self.SE2(Cnet2_out)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121

        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss
#-------------------------------------------10.17测试-------------------------#
class DoubleTFlearningCNet_FFTRDN_Transformer_ce(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_FFTRDN_Transformer_ce, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 2  # # 卷积层  数 原 3
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # RDN
        self.Constantnet1 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Constantnet2 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Transformer
        self.Transformerblock = TransformerBlock(dim=1).cuda()
        self.conv = torch.nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Attention
        self.SE1 = SEAttention(242, 16)
        self.SE2 = SEAttention(242, 16)
        self.conv1 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = input  ## 1个通道
        # # FFT RDN FFT # #
        input1_fft = torch.fft.fft2(input1).cuda() / 256  # (1 1 256 256)
        real = input1_fft.real # 1 122 256 256
        imag = input1_fft.imag # 1 122 256 256

        Constantnet_denseblock10 = self.Constantnet1(real)
        Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121

        Constantnet_denseblock101 = self.Constantnet1(imag)
        Constantnet_denseblock111 = self.Constantnet_denseblock11(Constantnet_denseblock101)
        Constantnet_denseblock121 = self.Constantnet_denseblock12(Constantnet_denseblock111)
        resnet_out11 = resnetblock(Constantnet_denseblock101, Constantnet_denseblock121)  ## 121+121 = 121
        out1 = torch.complex(resnet_out1, resnet_out11)  # 1 121 256 256 复数
        out1_ifft = torch.fft.ifft2(out1).cuda() * 256 # 1 121 256 256 复数
        out1_ifft = abs(out1_ifft)
        print(out1_ifft.shape)
        # # Transformer # #
        FSAS1 = self.Transformerblock(input1) # 1 122 256 256
        FSAS1_1 = self.conv(FSAS1) # 1 121 256 256
        print(FSAS1_1.shape)
        # # 级联
        Cnet1_out = torch.cat((out1_ifft, FSAS1_1), dim=1) #1 242 256 256
        # print(Cnet1_out.shape)
        Constantnet1 = self.SE1(Cnet1_out)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        ##########################################
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = input  #### 122

        input2_fft = torch.fft.fft2(input2).cuda() / 256  # (1 1 256 256)
        real2 = input2_fft.real  # 1 1 256 256
        imag2 = input2_fft.imag  # 1 1 256 256

        Constantnet_denseblock20 = self.Constantnet2(real2)
        Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)

        Constantnet_denseblock201 = self.Constantnet2(imag2)
        Constantnet_denseblock211 = self.Constantnet_denseblock21(Constantnet_denseblock201)
        Constantnet_denseblock221 = self.Constantnet_denseblock22(Constantnet_denseblock211)
        resnet_out21 = resnetblock(Constantnet_denseblock201, Constantnet_denseblock221)
        out2 = torch.complex(resnet_out2, resnet_out21)  # 1 1 256 256 复数
        out2_ifft = torch.fft.ifft2(out2).cuda() * 256  # 1 1 256 256 复数
        out2_ifft = abs(out2_ifft)
        # # Transformer # #
        FSAS2 = self.Transformerblock(input2)  # 1 122 256 256
        FSAS2_1 = self.conv(FSAS2)  # 1 121 256 256

        # # 级联
        Cnet2_out = torch.cat((out2_ifft, FSAS2_1), dim=1)

        Constantnet2 = self.SE2(Cnet2_out)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121

        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss

class DoubleTFlearningCNet_RDN_Transformer_nowx(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_RDN_Transformer_nowx, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数
        self.num_convs = 3  # # 卷积层  数 原 3
        self.in_channels = 121  # 11^2
        self.out_channels = 64
        self.in_channelss = self.in_channels + self.num_convs * self.out_channels
        # # RDN
        self.Constantnet1 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock11 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels, self.out_channels).cuda()
        self.Constantnet_denseblock12 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_1 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Constantnet2 = CNet(self.num_filters).cuda()
        self.Constantnet_denseblock21 = DenseBlock_conv_tong_cnet(self.num_convs, self.in_channels,self.out_channels).cuda()
        self.Constantnet_denseblock22 = transition_block_kernel_size3(self.in_channelss, self.num_filters).cuda()
        # self.conv_w3_3_2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Transformer
        self.Transformerblock = TransformerBlock(dim=1).cuda()
        self.conv = torch.nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        # # Attention
        self.SE1 = SEAttention(242, 16)
        self.SE2 = SEAttention(242, 16)
        self.conv1 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)

        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  ###这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  ### 输入1 输出 121
        input1 = input ## 1个通道
        # # FFT RDN FFT #
        Constantnet_denseblock10 = self.Constantnet1(input1)
        Constantnet_denseblock11 = self.Constantnet_denseblock11(Constantnet_denseblock10)
        Constantnet_denseblock12 = self.Constantnet_denseblock12(Constantnet_denseblock11)
        resnet_out1 = resnetblock(Constantnet_denseblock10, Constantnet_denseblock12)   ## 121+121 = 121

        # # Transformer # #
        FSAS1 = self.Transformerblock(input1) # 1 1 256 256
        FSAS1_1 = self.conv(FSAS1) # 1 121 256 256

        # # 级联
        Cnet1_out = torch.cat((resnet_out1, FSAS1_1), dim=1)
        Constantnet1 = self.SE1(Cnet1_out)
        Constantnet1 = self.conv1(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121

        ##########################################
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = input  #### 1

        Constantnet_denseblock20 = self.Constantnet2(input2)
        Constantnet_denseblock21 = self.Constantnet_denseblock21(Constantnet_denseblock20)
        Constantnet_denseblock22 = self.Constantnet_denseblock22(Constantnet_denseblock21)
        resnet_out2 = resnetblock(Constantnet_denseblock20, Constantnet_denseblock22)

        # # Transformer # #
        FSAS2 = self.Transformerblock(input2)  # 1 1 256 256
        FSAS2_1 = self.conv(FSAS2)  # 1 121 256 256

        # # 级联
        Cnet2_out = torch.cat((resnet_out2, FSAS2_1), dim=1)

        Constantnet2 = self.SE2(Cnet2_out)
        Constantnet2 = self.conv2(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121

        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size ** 2)
        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss
#---------------------------------10.18-----------------------------------#
class CNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu2 = nn.ReLU()
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu3 = nn.ReLU()
        self.conv4 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.relu1(out1)
        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.conv3(out4)
        out6 = self.relu3(out5)
        out = self.conv4(out6)

        return out
#-------------------pytorch 1.10 实部虚部分开6层--------------------#
class DoubleTFlearningCNet(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数

        # # CNN
        self.CNN1 = CNN(1, 121).cuda()
        self.CNN2 = CNN(1, 121).cuda()

        # # FSAS
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(64, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv4 = torch.nn.Conv2d(64, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.FSAS1 = TransformerBlock(dim=64)
        self.FSAS2 = TransformerBlock(dim=64)

        # # Attention
        self.SE1 = SEAttention(242, 16)
        self.SE2 = SEAttention(242, 16)
        self.conv5 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv6 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  # 这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  # 输入1 输出 121
        # input1 = torch.cat((input, Wx1), dim=1)  # 122个通道
        input1 = input  # 1 1 256 256
        # # fft CNN ifft
        fft_input1 = torch.fft.fft2(input1).cuda() / 256  # 1 1 256 256 复数
        real1 = fft_input1.real  # 1 1 256 256
        imag1 = fft_input1.imag  # 1 1 256 256
        cnn1_real = self.CNN1(real1)  # 1 121 256 256
        cnn1_imag = self.CNN1(imag1)
        out1 = torch.complex(cnn1_real, cnn1_imag)  # 1 121 256 256 复数
        ifft_out1 = torch.fft.ifft2(out1).cuda() * 256  # 1 121 256 256 复数
        ifft_out11 = ifft_out1.real
        # # conv FSAS conv
        conv1 = self.conv1(input1)
        fsas1 = self.FSAS1(conv1)
        conv2 = self.conv2(fsas1)  # 1 121 256 256
        # # 级联
        Cnet1_out = torch.cat((ifft_out11, conv2), dim=1)
        Constantnet1 = self.SE1(Cnet1_out)
        Constantnet1 = self.conv5(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        ############################################################################
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        # input2 = torch.cat((input, Wx2), dim=1)  # 122
        input2 = input
        # # fft CNN ifft
        fft_input2 = torch.fft.fft2(input2).cuda() / 256  # 1 1 256 256 复数
        real2 = fft_input2.real  # 1 1 256 256
        imag2 = fft_input2.imag  # 1 1 256 256
        cnn2_real = self.CNN2(real2)  # 1 121 256 256
        cnn2_imag = self.CNN2(imag2)
        out2 = torch.complex(cnn2_real, cnn2_imag)  # 1 121 256 256 复数
        ifft_out2 = torch.fft.ifft2(out2).cuda() * 256  # 1 121 256 256 复数
        ifft_out22 = ifft_out2.real
        # # conv FSAS conv
        conv3 = self.conv3(input2)
        fsas2 = self.FSAS2(conv3)
        conv4 = self.conv4(fsas2)  # 1 121 256 256
        # # 级联
        Cnet2_out = torch.cat((ifft_out22, conv4), dim=1)
        Constantnet2 = self.SE2(Cnet2_out)
        Constantnet2 = self.conv6(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121

        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size**2)

        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss

class DoubleTFlearningCNet_1025(nn.Module):
    def __init__(self):
        super(DoubleTFlearningCNet_1025, self).__init__()
        self.kernel_size = 11  # # 卷积核大小
        self.num_filters = self.kernel_size ** 2  # # 滤波器个数

        # # CNN
        self.CNN1 = CNN(244, 242).cuda()
        self.CNN2 = CNN(244, 242).cuda()

        # # FSAS
        self.conv1 = torch.nn.Conv2d(122, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv2 = torch.nn.Conv2d(64, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv3 = torch.nn.Conv2d(122, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv4 = torch.nn.Conv2d(64, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.FSAS1 = TransformerBlock(dim=64)
        self.FSAS2 = TransformerBlock(dim=64)

        # # Attention
        self.SE1 = SEAttention(242, 16)
        self.SE2 = SEAttention(242, 16)
        self.conv5 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.conv6 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)
        self.soft_threshold = SoftThreshold(self.num_filters, 1)
        self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
    def savefilters(self):
        rshape = self.conv_w2.weight.data.shape
        tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
        tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
        tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
        torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
        dshape = self.conv_w1.weight.data.shape
        tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
        tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
        tmax = tmax.view(dshape[0], dshape[1], 1, 1)
        tmin = tmin.view(dshape[0], dshape[1], 1, 1)
        torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
    def setdecfilter(self):
        mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
        tframedict = mat_dict['learnt_dict'].transpose()
        self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
    def forward(self, input, sigma_map):  # 这里的sigma已经是一个噪声图了
        Wx1 = self.conv_w1(self.rpad(input))  # 输入1 输出 121
        input1 = torch.cat((input, Wx1), dim=1)  # 122个通道
        # input1 = input  # 1 1 256 256
        # # fft CNN ifft
        fft_input1 = torch.fft.fft2(input1).cuda() / 256  # 1 1 256 256 复数
        fft_input1 = torch.cat((fft_input1.real, fft_input1.imag), dim=1) # 1 244 256 256
        cnn1 = self.CNN1(fft_input1)  # 1 242 256 256
        out1 = torch.complex(cnn1[:, 0:121, :, :], cnn1[:, 121:, :, :])  # 1 121 256 256 复数
        ifft_out1 = torch.fft.ifft2(out1).cuda() * 256  # 1 121 256 256 复数
        ifft_out11 = ifft_out1.real  # 1 121 256 256
        # # conv FSAS conv
        conv1 = self.conv1(input1)
        fsas1 = self.FSAS1(conv1)
        conv2 = self.conv2(fsas1)  # 1 121 256 256
        # # 级联
        Cnet1_out = torch.cat((ifft_out11, conv2), dim=1)
        Constantnet1 = self.SE1(Cnet1_out)
        Constantnet1 = self.conv5(Constantnet1)
        constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
        epsilon_hat1 = constant1 * sigma_map
        z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
        ############################################################################
        Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
        input2 = torch.cat((input, Wx2), dim=1)  # 122
        # input2 = input
        # # fft CNN ifft
        fft_input2 = torch.fft.fft2(input2).cuda() / 256  # 1 1 256 256 复数
        fft_input2 = torch.cat((fft_input2.real, fft_input2.imag), dim=1)  # 1 2 256 256
        cnn2 = self.CNN2(fft_input2)  # 1 242 256 256
        out2 = torch.complex(cnn2[:, 0:121, :, :], cnn2[:, 121:, :, :])  # 1 121 256 256 复数
        ifft_out2 = torch.fft.ifft2(out2).cuda() * 256  # 1 121 256 256 复数
        ifft_out22 = ifft_out2.real  # 1 121 256 256
        # # conv FSAS conv
        conv3 = self.conv3(input2)
        fsas2 = self.FSAS2(conv3)
        conv4 = self.conv4(fsas2)  # 1 121 256 256
        # # 级联
        Cnet2_out = torch.cat((ifft_out22, conv4), dim=1)
        Constantnet2 = self.SE2(Cnet2_out)
        Constantnet2 = self.conv6(Constantnet2)
        constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
        epsilon_hat2 = constant2 * sigma_map
        z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121

        # # inverse transform
        weightT1 = rot180(self.conv_w1.weight)
        weightT2 = rot180(self.conv_w2.weight)
        Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size**2)

        # # W2 loss
        temp = self.conv_w2(self.rpad(Wx1))
        WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
        W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)

        return Wt, self.conv_w1.weight, W2loss
#------------------pytorch 1.13 复数卷积(暂时行不通)--------------------------#
# class DoubleTFlearningCNet2(nn.Module):
#     def __init__(self):
#         super(DoubleTFlearningCNet2, self).__init__()
#         self.kernel_size = 11  # # 卷积核大小
#         self.num_filters = self.kernel_size ** 2  # # 滤波器个数
#
#         # # CNN
#         self.CNN1 = CNN(1, 121).cuda()
#         self.CNN2 = CNN(1, 121).cuda()
#
#         # # FSAS
#         self.conv1 = torch.nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.conv2 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.conv3 = torch.nn.Conv2d(1, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.conv4 = torch.nn.Conv2d(121, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.FSAS1 = TransformerBlock(dim=121)
#         self.FSAS2 = TransformerBlock(dim=121)
#
#         # # Attention
#         self.SE1 = SEAttention(242, 16)
#         self.SE2 = SEAttention(242, 16)
#         self.conv5 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#         self.conv6 = torch.nn.Conv2d(242, 121, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
#
#         self.conv_w1 = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0)
#         self.conv_w2 = nn.Conv2d(self.num_filters, self.num_filters, self.kernel_size, stride=1, padding=0)
#         self.soft_threshold = SoftThreshold(self.num_filters, 1)
#         self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)   # 填充，但通道数没变
#     def savefilters(self):
#         rshape = self.conv_w2.weight.data.shape
#         tmax, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).max(2)
#         tmin, _ = self.conv_w2.weight.data.view(rshape[0], rshape[1], -1).min(2)
#         tmax = tmax.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
#         tmin = tmin.permute(1, 0).view(rshape[1], rshape[0], 1, 1)
#         torchvision.utils.save_image((self.conv_w2.weight.data.permute(1, 0, 2, 3) - tmin) / (1e-10 + tmax - tmin), 'recfilter.png', nrow=self.kernel_size, padding=1)
#         dshape = self.conv_w1.weight.data.shape
#         tmax, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).max(2)
#         tmin, _ = self.conv_w1.weight.data.view(dshape[0], dshape[1], -1).min(2)
#         tmax = tmax.view(dshape[0], dshape[1], 1, 1)
#         tmin = tmin.view(dshape[0], dshape[1], 1, 1)
#         torchvision.utils.save_image((self.conv_w1.weight.data - tmin) / (1e-10 + tmax - tmin), 'decfilter.png', nrow=self.kernel_size, padding=1)
#     def setdecfilter(self):
#         mat_dict = scipy.io.loadmat('DoubleTightFrames/globalTF11.mat')
#         tframedict = mat_dict['learnt_dict'].transpose()
#         self.conv_w1.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w1.weight.shape).permute(0, 1, 3, 2).contiguous()
#     def forward(self, input, sigma_map):  # 这里的sigma已经是一个噪声图了
#         Wx1 = self.conv_w1(self.rpad(input))  # 输入1 输出 121
#         # input1 = torch.cat((input, Wx1), dim=1)  # 122个通道
#         input1 = input  # 1 1 256 256
#         # # fft CNN ifft
#         fft_input1 = torch.fft.fft2(input1).cuda() / 256  # 1 121 256 256 复数
#         cnn1 = self.CNN1(fft_input1)  # 1 122 256 256
#         ifft_out1 = torch.fft.ifft2(cnn1).cuda() * 256 # 1 121 256 256 复数
#         ifft_out1 = ifft_out1.real
#         # # conv FSAS conv
#         conv1 = self.conv1(input1)
#         fsas1 = self.FSAS1(conv1)
#         conv2 = self.conv2(fsas1)  # 1 121 256 256
#         # # 级联
#         Cnet1_out = torch.cat((ifft_out1, conv2), dim=1)
#         Constantnet1 = self.SE1(Cnet1_out)
#         Constantnet1 = self.conv5(Constantnet1)
#         constant1 = torch.clamp(Constantnet1, 0., 10.)  # 0<C<10
#         epsilon_hat1 = constant1 * sigma_map
#         z1 = self.soft_threshold(Wx1, epsilon_hat1)  ### 输入121 输出 121
#         ############################################################################
#         Wx2 = self.conv_w2(self.rpad(z1))  ### 输入121 输出121
#         # input2 = torch.cat((input, Wx2), dim=1)  # 122
#         input2 = input
#         # # fft CNN ifft
#         fft_input2 = torch.fft.fft2(input2).cuda() / 256  # 1 1 256 256 复数
#         cnn2 = self.CNN2(fft_input2)  # 1 121 256 256
#         ifft_out2 = torch.fft.ifft2(cnn2).cuda() * 256  # 1 121 256 256 复数
#         ifft_out2 = ifft_out2.real
#         # # conv FSAS conv
#         conv3 = self.conv3(input2)
#         fsas2 = self.FSAS2(conv3)
#         conv4 = self.conv4(fsas2)  # 1 121 256 256
#         # # 级联
#         Cnet2_out = torch.cat((ifft_out2, conv4), dim=1)
#         Constantnet2 = self.SE2(Cnet2_out)
#         Constantnet2 = self.conv6(Constantnet2)
#         constant2 = torch.clamp(Constantnet2, 0., 10.)  # 0<C<10
#         epsilon_hat2 = constant2 * sigma_map
#         z2 = self.soft_threshold(Wx2, epsilon_hat2)  ###输出121
#
#         # # inverse transform
#         weightT1 = rot180(self.conv_w1.weight)
#         weightT2 = rot180(self.conv_w2.weight)
#         Wt2 = F.conv2d(self.rpad(z2), weightT2, stride=1, padding=0)/(self.kernel_size**2)
#         Wt = F.conv2d(self.rpad(Wt2), weightT1, stride=1, padding=0)/(self.kernel_size**2)
#
#         # # W2 loss
#         temp = self.conv_w2(self.rpad(Wx1))
#         WTW2 = F.conv2d(self.rpad(temp), weightT2, stride=1, padding=0)/(self.kernel_size**2)
#         W2loss = nn.functional.mse_loss(WTW2, Wx1, size_average=False, reduce=True)
#
#         return Wt, self.conv_w1.weight, W2loss

class TNet_CNN_1_2(nn.Module):
    def __init__(self):
        super(TNet_CNN_1_2, self).__init__()
        featuremap= 121
        self.apply_A = torch.nn.Conv2d(121, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_B = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_C = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_D = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_E = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_F = torch.nn.Conv2d(featuremap, 121, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        temp1 = self.relu(self.apply_A(x))
        temp2 = self.relu(self.apply_B(temp1))
        temp3 = self.relu(self.apply_C(temp2))
        temp4 = self.relu(self.apply_D(temp3))
        temp5 = self.relu(self.apply_E(temp4))
        y = self.apply_F(temp5)
        return y

class TNet_CNN_3_6(nn.Module):
    def __init__(self):
        super(TNet_CNN_3_6, self).__init__()
        featuremap= 121
        self.apply_A = torch.nn.Conv2d(121, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_B = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_C = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_D = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_E = torch.nn.Conv2d(featuremap, featuremap, kernel_size=3, stride=1, padding=1)
        self.apply_F = torch.nn.Conv2d(featuremap, 121, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        temp1 = self.relu(self.apply_A(x))
        temp2 = self.relu(self.apply_B(temp1))
        temp3 = self.relu(self.apply_C(temp2))
        temp4 = self.relu(self.apply_D(temp3))
        temp5 = self.relu(self.apply_E(temp4))
        y = self.apply_F(temp5)
        return y

def PhiTPhi_fun(x, PhiW):
    temp = F.conv2d(x, PhiW, padding=0, stride=32, bias=None)
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# Define Cross Attention Block
class blockNL(torch.nn.Module):
    def __init__(self, channels):
        super(blockNL, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)

        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)
        self.v = nn.Conv2d(in_channels=self.channels + 1, out_channels=self.channels + 1, kernel_size=1, stride=1,
                           bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False,
                      groups=self.channels),
        )

    def forward(self, x, z):
        x0 = self.norm_x(x)
        z0 = self.norm_z(z)

        z1 = self.t(z0)
        b, c, h, w = z1.shape
        z1 = z1.view(b, c, -1)  # b, c, hw
        x1 = self.p(x0)  # b, c, hw
        x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x_t = x1.permute(0, 2, 1)  # b, hw, c
        att = torch.matmul(z1, x_t)
        att = self.softmax(att)  # b, c, c

        z2 = self.g(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + self.pos_emb(z2) + z
        y = self.v(torch.cat([x, out_x], 1))

        return y


# Define ISCA block
class Atten(torch.nn.Module):
    def __init__(self, channels):
        super(Atten, self).__init__()

        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.norm1 = LayerNorm(self.channels, 'WithBias')
        self.norm2 = LayerNorm(self.channels, 'WithBias')
        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.conv_kv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels * 2, self.channels * 2, kernel_size=3, stride=1, padding=1,
                      groups=self.channels * 2, bias=True)
        )
        self.conv_out = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
                                  bias=True)

    def forward(self, pre, cur):
        b, c, h, w = pre.shape
        pre_ln = self.norm1(pre)
        cur_ln = self.norm2(cur)
        q = self.conv_q(cur_ln)
        q = q.view(b, c, -1)
        k, v = self.conv_kv(pre_ln).chunk(2, dim=1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        att = torch.matmul(q, k.permute(0, 2, 1))
        att = self.softmax(att)
        out = torch.matmul(att, v).view(b, c, h, w)
        out = self.conv_out(out) + cur

        return out


# Define OCT
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.atten = Atten(31)
        self.nonlo = blockNL(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )

    def forward(self, x, z_pre, z_cur):
        z = self.atten(z_pre, z_cur)
        # x_input = self.d(x, FB, FBC, F2B, FBFy, ab, sf)
        x_input = x

        x_input = self.nonlo(x_input, z)

        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward
        x_pred = x_input + x_backward

        return x_pred

# Define OCTUF
class TNet_SCT(torch.nn.Module):
    def __init__(self):
        super(TNet_SCT, self).__init__()
        onelayer = []
        self.LayerNo = 3
        self.patch_size = 32

        for i in range(self.LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x_t, x_t_2, x_t_1):
        x = x_t
        z_pre = x_t_2
        z_cur = x_t_1

        for i in range(self.LayerNo):
            x_dual = self.fcs[i](x, z_pre, z_cur)
            x = x_dual[:, :1, :, :]
            z_pre = z_cur
            z_cur = x_dual[:, 1:, :, :]

        # x_final = x
        x_final = x_dual

        return x_final

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# class SCTNet_1(nn.Module):  # 单层紧标架网络
#     def __init__(self):
#         super(SCTNet_1, self).__init__()
#         self.kernel_size = 11
#         self.num_filters = self.kernel_size ** 2
#         self.conv_w = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0, bias=False)  ### 卷积不做padding
#         self.soft_threshold = SoftThreshold(self.num_filters, 1)
#         self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充# weight initial
#         self.TNet_CNN = TNet_CNN_3_6()
#         self.TNet_SCT = TNet_SCT()
#         self.apply_A = torch.nn.Conv2d(1, 31, kernel_size=3, stride=1, padding=1)
#         self.apply_B = torch.nn.Conv2d(1, 31, kernel_size=3, stride=1, padding=1)
#         self.apply_C = torch.nn.Conv2d(153, 121, kernel_size=3, stride=1, padding=1)
#         self.atten = CALayer(153)
#         # self.tau = 0.5
#         self.tau = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#         self.lamb = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#
#     # def setdecfilter(self):
#     #     # mat_dict = scipy.io.loadmat('D:/xwy/LearnDnCNN_0616_2e-3/globalTF9.mat')
#     #     mat_dict = scipy.io.loadmat('globalTF11.mat')
#     #     tframedict = mat_dict['learnt_dict'].transpose()
#     #     self.conv_w.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w.weight.shape).permute(0, 1, 3,
#     #                                                                                                      2).contiguous()
#     def forward(self, input):
#         Wx = self.conv_w(self.rpad(input))  # conv_w：紧标架
#
#         epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#         x_0_in = self.apply_A(input)
#         x_1_in = self.apply_B(input)
#         epsilon_2 = self.TNet_SCT(input, x_0_in, x_1_in)
#         epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#         epsilon = self.atten(epsilon)
#         epsilon = self.apply_C(epsilon)
#         epsilon = torch.clamp(epsilon, 1e-5, 10)
#
#         z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#         weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#         Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#         temp = Wt
#         weight = self.conv_w.weight
#         out = (1 - self.tau * self.lamb) * input + self.tau * self.lamb * temp
#         # return out, weight
#
#         Wx = self.conv_w(self.rpad(input))  # conv_w：紧标架
#         epsilon = self.TNet_CNN_6(Wx)  # 计算epsilon
#         epsilon = torch.clamp(epsilon, 1e-5, 10)
#
#         z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#         weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#         Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#         temp = Wt
#         weight = self.conv_w.weight
#         out = (1 - self.tau * self.lamb) * input + self.tau * self.lamb * temp
#
#         return out, weight
#
#
# class SCTNet_2(nn.Module):  # 单层紧标架网络
#     def __init__(self):
#         super(SCTNet_2, self).__init__()
#         self.kernel_size = 11
#         self.num_filters = self.kernel_size ** 2
#         self.conv_w = nn.Conv2d(1, self.num_filters, self.kernel_size, stride=1, padding=0, bias=False)  ### 卷积不做padding
#         self.soft_threshold = SoftThreshold(self.num_filters, 1)
#         self.rpad = nn.ReflectionPad2d(self.kernel_size // 2)  # 对输入图像以最外围像素为对称轴，做四周的轴对称镜像填充# weight initial
#         self.TNet_CNN = TNet_CNN_3_6()
#         self.TNet_SCT = TNet_SCT()
#         self.apply_A = torch.nn.Conv2d(1, 31, kernel_size=3, stride=1, padding=1)
#         self.apply_B = torch.nn.Conv2d(1, 31, kernel_size=3, stride=1, padding=1)
#         self.apply_C = torch.nn.Conv2d(153, 121, kernel_size=3, stride=1, padding=1)
#         self.atten = CALayer(153)
#         # self.tau = 0.5
#         self.tau = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#         self.lamb = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#
#     # def setdecfilter(self):
#     #     # mat_dict = scipy.io.loadmat('D:/xwy/LearnDnCNN_0616_2e-3/globalTF9.mat')
#     #     mat_dict = scipy.io.loadmat('globalTF11.mat')
#     #     tframedict = mat_dict['learnt_dict'].transpose()
#     #     self.conv_w.weight.data = torch.Tensor(tframedict).cuda().view(self.conv_w.weight.shape).permute(0, 1, 3,
#     #                                                                                                      2).contiguous()
#     def forward(self, input):
#         Wx = self.conv_w(self.rpad(input))  # conv_w：紧标架
#
#         epsilon_1 = self.TNet_CNN(Wx)  # 计算epsilon
#         x_0_in = self.apply_A(input)
#         x_1_in = self.apply_B(input)
#         epsilon_2 = self.TNet_SCT(input, x_0_in, x_1_in)
#         epsilon = torch.cat((epsilon_1, epsilon_2), 1)
#         epsilon = self.atten(epsilon)
#         epsilon = self.apply_C(epsilon)
#         epsilon = torch.clamp(epsilon, 1e-5, 10)
#
#         z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#         weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#         Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#         temp = Wt
#         weight = self.conv_w.weight
#         out = (1 - self.tau * self.lamb) * input + self.tau * self.lamb * temp
#         # return out, weight
#
#         Wx = self.conv_w(self.rpad(input))  # conv_w：紧标架
#         epsilon = self.TNet_CNN_6(Wx)  # 计算epsilon
#         epsilon = torch.clamp(epsilon, 1e-5, 10)
#
#         z = self.soft_threshold(Wx, epsilon)  # soft操作，需要epsilon作为输入
#         weightT = rot180(self.conv_w.weight)  ###  输出通道数 输入通道数  滤波器大小  W的转置
#         Wt = F.conv2d(self.rpad(z), weightT, stride=1, padding=0) / (self.kernel_size ** 2)  # 7*7
#         temp = Wt
#         weight = self.conv_w.weight
#         out = (1 - self.tau * self.lamb) * input + self.tau * self.lamb * temp
#
#         return out, weight
#
