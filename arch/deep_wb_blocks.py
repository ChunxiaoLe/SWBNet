"""
 The main blocks of SWBNet

 Reference：
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""

import torch.nn as nn
from .vit import *

import torch.nn.functional as F
from .zigzag import *
import math

class DoubleConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels, B=False):
        super().__init__()
        if B:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)

class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels,B=B)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class BridgeDown(nn.Module):
    """Downscale bottleneck block: maxpool -> conv"""
    def __init__(self, in_channels, out_channels,B):
        super().__init__()
        if B:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )

        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_up = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.conv_up(x)

class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels * 2, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)



    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))

class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.Sequential(
            DoubleConvBlock(in_channels * 2, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)

class DownBlock_res_dct1(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels,att=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.dct_layer = DCT_layer(channel=in_channels,N=8,idx=[0,1,2,5])
        self.conv_sp = DoubleConvBlock(in_channels, out_channels)

        self.a = att
        if self.a:
            self.att = ct_removing_block(in_channels)

    def forward(self, x):
        x_ = self.maxpool(x)
        x_va,x_in = self.dct_layer(x_)
        if self.a:
            x_va_,gamma = self.att(x_va)
            x_all = x_va_ + x_in
            x_out = self.conv_sp(x_all)
            return x_out
        else:
            x_all = torch.cat((x_va, x_in), 1)
            x_out = self.conv_sp(x_all)
            return x_out

class DCT_layer(nn.Module):
    def __init__(self, channel,N,idx):
        super().__init__()

        self.N = N
        self.idx = idx
        self.register_buffer('kernel_dct', basis_func_initial(N,channel))
        self.kernel_dct = self.kernel_dct[:, self.idx, :, :].cuda('cuda:1')

        self.k = len(self.idx)

        self.kernel_dct_ = torch.reshape(self.kernel_dct, (-1, 1, N, N))

        self.kernel_dctd_ = self.flip(self.kernel_dct,2)
        self.kernel_dctd_ = self.flip(self.kernel_dctd_, 3)

        self.bise_dct = None
        self.stride = N
        self.channel = channel

        self.up = nn.ConvTranspose2d(channel, channel, kernel_size=8)

        self.p1 = int((self.N)/2)
        self.p2 = int((self.N)/2) - 1
        self.padding = nn.ReplicationPad2d((self.p1,self.p2,self.p1,self.p2))


    def forward(self, x):
        x_ = x
        b1,c1,h1,w1 = x_.size()
        x1 = x.repeat_interleave(self.k,dim=1)
        b,c,h,w = x1.size()

        x_dct = F.conv2d(x1, self.kernel_dct_, self.bise_dct, stride=self.stride,
                      padding=0,groups=c)

        y = self.padding_zeros(x1,x_dct)

        y1 = F.conv2d(y, self.kernel_dctd_, self.bise_dct, stride=1, padding=0, groups=self.channel)

        y1 = torch.nn.functional.interpolate(y1, [h1, w1])

        y2_ = x_ - y1
        return y1,y2_

    def init_cos(self):
        A = torch.zeros((1, 1, self.N, self.N))
        A[0, 0, 0, :] = 1 * math.sqrt(1 / self.N)
        for i in range(1, self.N):
            for j in range(self.N):
                A[0, 0, i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * self.N)
                                         ) * math.sqrt(2 / self.N)
        return A

    def flip(self, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                     -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

    def padding_zeros(self, x1, x_dct):

        _, _, h, w = x1.shape
        _, _, h_dct, w_dct = x_dct.shape
        y = torch.zeros_like(x1)
        idx_h = torch.arange(self.N - 1, h, self.N).unsqueeze(0)
        h_idx = idx_h.repeat_interleave(int(w / self.N), dim=1)

        idx_w = torch.arange(self.N - 1, w, self.N).unsqueeze(0)
        w_idx = idx_w.repeat_interleave(int(h / self.N), dim=0).long()

        h_idx = torch.reshape(h_idx, w_idx.shape).long()
        y[:, :, h_idx, w_idx] = x_dct

        return y


def basis_func_initial(N,Ch):
    """
    input: DCT basis martix size
    output: the initialzed basis function
    the output is used for the initialization of dct_conv
    out: group, N*N, N,N
    """
    #### 生成 N*N 个 N*N的基函数
    B = torch.zeros((Ch, N*N, N, N))
    ### N = 8
    B_ = torch.tensor(
        [[0, 1, 5, 6, 14, 15, 27,28], [2, 4, 7, 13, 16, 26, 29,42], [3, 8, 12, 17, 25, 30,41,43], [9, 11, 18, 24, 31, 40, 44,53],
         [10, 19, 23, 32, 39, 45, 52,54], [20, 22, 33,38,46,51,55,60], [21, 34,37,47,50,56,59,61],[35,36,48,49,57,58,62,63]])

    for i in range(N*N):
        #### i 对应的坐标
        i_ = torch.where(B_==i)
        # print(i)
        # print(i_)
        k1 = i_[0][0]
        k2 = i_[1][0]

        # print(k1, k2)
        if k1 == 0:
            a1 = math.sqrt(1)
        else:
            a1 = math.sqrt(2)

        if k2 == 0:
            a2 = math.sqrt(1)
        else:
            a2 = math.sqrt(2)

        for p1 in range(N):
            for p2 in range(N):
                B[:, i, p1, p2] = (1/N) * a1 * a2 * math.cos(((math.pi * (p1 + 0.5)) / N) * k1) * math.cos(
                    ((math.pi * (p2 + 0.5)) / N) * k2)
    return B


class ct_removing_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels,in_channels, kernel_size=1, stride=1)

        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
            x_ = x
            "gamma"
            avg = self.avg_pool(x)
            var = torch.var(x, dim=(2, 3), keepdim=True)
            gamma = self.fc2(self.relu1(self.fc1(avg + var)))
            "multiplying"
            xc_ = torch.mul(x_, gamma)
            xc_ = self.relu(self.conv(xc_))
            return xc_,gamma


class DownBlock_res_dct3(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels,att=True):
        super().__init__()
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        # self.dct_layer1 = DCT_layer(channel=in_channels, N=8, idx=64)
        self.dct_layer = DCT_layer2(channel=in_channels,N=8,idx=[0,1,2])
        # self.avgpool = nn.AvgPool2d(2)
        self.conv_sp = DoubleConvBlock(in_channels, out_channels)

        self.a = att
        if self.a:
            # self.att = SE_net(out_channels, out_channels, reduction=4, attention=att)
            # self.att = NonLocalBlock(channel=out_channels)
            # self.att = ct_removing_block(in_channels)
            self.att = ChannelAttention(in_planes=out_channels)

    def forward(self, x):
        # x_ = self.maxpool(x)
        x_va, x_in = self.dct_layer(x)
        if self.a:
            x_va_ = self.att(x_va)
            # x_all = x_va_ + x_in
            x_all = torch.cat((x_va_, x_in), 1)
            x_out = self.conv_sp(x_va_)
            return x_out
        else:
            # x_all = torch.cat((x_va, x_in), 1)
            x_out = self.conv_sp(x)
            return x_out



class CTSTRANS(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, hight, wide, patch_size,lay,heads):
        super().__init__()
        ### downsample
        self.down = nn.MaxPool2d(2)
        self.lay = lay
        channel = [3,12,24]
        self.ch = channel[self.lay]
        self.dim = self.ch * patch_size * patch_size

        if self.lay == 0:
            h_r2 = hight
            w_r2 = wide
        else:
            h_r2= hight // (self.lay*2)
            w_r2= wide // (self.lay*2)

        self.in_size = h_r2 * w_r2 * self.ch // self.dim

        self.out_dim = (3 * hight * wide) // self.in_size

        ### brand1
        if self.lay == 2:
            # self.maxpool = nn.MaxPool2d(2)
            self.conv_b1 = DoubleConvBlock(in_channels, 12)
            self.dct_layer1 = DCT_layer2(channel=in_channels, N=8, idx=[0, 1, 2,5])
            self.conv_b2 = DoubleConvBlock(12, 24)
            self.dct_layer2 = DCT_layer2(channel=12, N=8, idx=[0, 1, 2,5])
            # self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=9, padding=4, stride=1)

        elif self.lay == 1:
            self.conv_b1 = DoubleConvBlock(in_channels, 12)
            self.out_conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=9, padding=4, stride=1)

        elif self.lay == 0:
            self.out_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=9, padding=4, stride=1)

        # self.patch_transformer = ViT(image_size=h_r1, patch_size=patch_size, in_channel=12, dim=self.dim1,
        #                              depth=1, heads=4, mlp_dim=1024, dim_head=int(self.dim1 // 4))
        ### 默认为16
        heads = heads
        if self.lay == 0:
            dim_head = self.dim
        else:
            dim_head = int(self.dim//(self.lay*2))

        self.patch_transformer = ViT(image_size=(h_r2,w_r2), patch_size=patch_size,in_channel=self.ch, dim=self.dim,
                                     depth=1, heads=heads,mlp_dim=1024,dim_head=dim_head)


        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim*2)
        )

        #### 原先是self.dim * 2

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(self.dim*2),
            nn.Linear(self.dim*2, self.dim*2)
        )


        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        b,c,h,w = x.shape
        x_ = x
        if self.lay == 2:
            # x1_ = self.maxpool(x)
            # x1 = self.down(x)
            x1_, _ = self.dct_layer1(x_)
            x1 = self.conv_b1(x1_)
            # x1_, _ = self.dct_layer1(x1)
            # x2 = self.down(x1)
            x2_, _ = self.dct_layer2(x1)
            x2 = self.conv_b2(x2_)
            # x2_,_ = self.dct_layer2(x2)

            ### brand1
            att = self.patch_transformer(x2)
            # x_att2 = torch.mul(x_att1,att2)
            att = self.mlp_head1(att)
            att = self.mlp_head2(att)
            att = att.reshape(b,h,w,-1).permute(0,3,1,2).contiguous()
            # att = self.re_back(att)
            att = self.soft(self.out_conv(att))
            x_out = torch.mul(x_,att)
            return x_out,att

        elif self.lay == 1:
            x = self.down(x)
            x = self.conv_b1(x)

            # x = self.down(x)
            # x = self.conv_b2(x)
            ### brand1
            att = self.patch_transformer(x)
            # x_att2 = torch.mul(x_att1,att2)
            att = self.mlp_head1(att)
            att = self.mlp_head2(att)
            att = att.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            # att = self.re_back(att)
            att = self.soft(self.out_conv(att))

            # x_out = att + x_

            x_out = torch.mul(x_, att)

            return x_out, att

        elif self.lay == 0:
            ### brand1
            att = self.patch_transformer(x)
            # x_att2 = torch.mul(x_att1,att2)
            att = self.mlp_head1(att)
            att = self.mlp_head2(att)
            att = att.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
            # att = self.re_back(att)
            att = self.soft(self.out_conv(att))

            x_out = torch.mul(x_, att)

            return x_out, att


class DCT_layer2(nn.Module):
    def __init__(self, channel,N,idx):
        super().__init__()

        self.N = N
        self.idx = idx

        self.register_buffer('kernel_dct', basis_func_initial(N,channel))
        self.kernel_dct = self.kernel_dct[:, self.idx, :, :].cuda('cuda:1')

        self.k = len(self.idx)

        self.kernel_dct_ = torch.reshape(self.kernel_dct, (-1, 1, N, N))

        self.kernel_dctd_ = self.flip(self.kernel_dct,2)
        self.kernel_dctd_ = self.flip(self.kernel_dctd_, 3)

        self.bise_dct = None
        self.stride = N
        self.channel = channel

        self.up = nn.ConvTranspose2d(channel, channel, kernel_size=8)

        self.p1 = int((self.N)/2)
        self.p2 = int((self.N)/2) - 1
        self.padding = nn.ReplicationPad2d((self.p1,self.p2,self.p1,self.p2))

        self.maxpool = nn.MaxPool2d(2)


    def forward(self, x):
        x1 = x.repeat_interleave(self.k,dim=1)
        b,c,h,w = x1.size()

        x_d = self.maxpool(x)

        x_dct = F.conv2d(x1, self.kernel_dct_, self.bise_dct, stride=self.stride,
                      padding=0,groups=c)


        y = self.padding_zeros(x1,x_dct)

        y1 = F.conv2d(y, self.kernel_dctd_, self.bise_dct, stride=2, padding=0, groups=self.channel)
        y1 = self.padding(y1)
        # y1 = torch.nn.functional.interpolate(y1, [h1, w1])


        y2_ = x_d - y1
        return y1, y2_



    def init_cos(self):
        A = torch.zeros((1, 1, self.N, self.N))
        A[0, 0, 0, :] = 1 * math.sqrt(1 / self.N)
        for i in range(1, self.N):
            for j in range(self.N):
                A[0, 0, i, j] = math.cos(math.pi * i * (2 * j + 1) / (2 * self.N)
                                         ) * math.sqrt(2 / self.N)
        return A

    def flip(self, x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.view(-1, *xsize[dim:])
        x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1,
                                                                     -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)

    def padding_zeros(self,x1,x_dct):

        _,_,h,w = x1.shape
        _, _, h_dct, w_dct = x_dct.shape
        y = torch.zeros_like(x1)
        idx_h = torch.arange(self.N - 1, h, self.N).unsqueeze(0)
        h_idx = idx_h.repeat_interleave(int(w / self.N), dim=1)

        idx_w = torch.arange(self.N-1,w,self.N).unsqueeze(0)
        w_idx = idx_w.repeat_interleave(int(h / self.N), dim=0).long()

        h_idx = torch.reshape(h_idx, w_idx.shape).long()
        y[:, :, h_idx, w_idx] = x_dct

        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # kk = torch.var(x,dim=(2,3),keepdim=True)
        # kk1 = torch.var(kk,dim=3,keepdim=True)
        var_out = self.fc2(self.relu1(self.fc1(torch.var(x,dim=(2,3),keepdim=True))))

        w = self.sigmoid(avg_out + var_out)


        # out = max_out
        return x*w

