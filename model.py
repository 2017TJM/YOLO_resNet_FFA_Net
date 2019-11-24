import torch.nn as nn
import torch

import torch.nn.functional as F
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
import math #数学模块

# class PALayer(nn.Module):
#     def __init__(self, channel):
#         super(PALayer, self).__init__()
#         self.pa = nn.Sequential(
#             nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
    def forward(self, x):
        y = self.pa(x)
        print("y.shape",y.shape) #2,1,14,14
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)  #32,448,448
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res
class Block_2(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_2, self).__init__()
        self.conv1 = conv(128, 128, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(128, 128, kernel_size, bias=True)
        self.calayer = CALayer(128)
        self.palayer = PALayer(128)

    def forward(self, x):
        res = self.act1(self.conv1(x))
       # print("res.shape",res.shape) #2,128,56,56
        res = res + x
        res = self.conv2(res)  #32,448,448
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
       # print("res____2222",res.shape) #2,128,56,56
        return res

class Block_3(nn.Module):
    def __init__(self, conv, dim, kernel_size, ):
        super(Block_3, self).__init__()
        self.conv1 = conv(128, 128, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(128, 128, kernel_size, bias=True)
        self.calayer = CALayer(128)
        self.palayer = PALayer(128)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        print("res3.shape__",res.shape) #2,128,56,56
        res = res + x
        res = self.conv2(res)  #32,448,448
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
       # print("res____2222",res.shape) #2,128,56,56
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks): #2,128,56,56
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
        self.Conv_1 =nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1,bias=True)
        self.bn =nn.BatchNorm2d(64)
        self.Block_res =Block_res(64,64)
        self.Conv_2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn_2 =nn.BatchNorm2d(128)
    def forward(self, x):
        res = self.gp(x)
        res += x
        res_1 =self.Conv_1(res)
        res =self.bn(res_1)
        res =self.Block_res(res)
        res =self.Conv_2(res)
        res =self.bn_2(res)
        return res
class Group_2(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks): #2,128,56,56
        super(Group_2, self).__init__()
        modules = [Block_3(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(128, 128, kernel_size))
        self.gp = nn.Sequential(*modules)
        self.Conv_1 =nn.Conv2d(128,128,kernel_size=3,stride=2,padding=1,bias=True)
        self.bn =nn.BatchNorm2d(128)
        self.Conv_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=True)
        self.Block_res =Block_res(128,128)       #
    def forward(self, x): #-1,128,56,56
        res = self.gp(x) #2,32,56,56
       # print("res_gp_shape",res.shape) #
        res += x
        res_1 =self.Conv_1(res)
        res =self.bn(res_1)
        res =self.Block_res(res)
        res_2 =self.Conv_2(res)
        res_2 = self.bn(res_2)
     #   print("res_2",res_2.shape) #2,128,14,14->
        return res_2
class Group_3(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks): #2,128,56,56
        super(Group_3, self).__init__()
        modules = [Block_3(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(128, 128, kernel_size))
        self.gp = nn.Sequential(*modules)
        self.Conv_1 =nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn =nn.BatchNorm2d(128)
        self.Conv_2=nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=True)
        self.Block_res =Block_res(128,128,stride_change=True)       #
    def forward(self, x): #-1,128,56,56
        res = self.gp(x) #2,32,56,56
        #print("res_gp_shape",res.shape) #2,128,14,14
        res += x
        res_1 =self.Conv_1(res)
       # print("res_1__3",res_1.shape) #2,128,14,14
        res =self.bn(res_1)
        res =self.Block_res(res)
        res_3 =self.Conv_2(res)
        res_3 =self.bn(res_3)
        return res_3

class FFA(nn.Module):
    def __init__(self, gps, blocks, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 32
        kernel_size = 3
        self.dim_ =128
        pre_process = [conv(3, self.dim, kernel_size)] #进行一次前置卷积->32，448，448
      #  assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group_2(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group_3(conv, self.dim, kernel_size, blocks=blocks)
        self.avg_res1 = nn.AdaptiveAvgPool2d(14)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])
        self.palayer = PALayer(11)

        # post_precess = [                             #首先卷积一次，通道数为32
        #     conv(self.dim, self.dim, kernel_size),
        #     conv(self.dim, 3, kernel_size)]
        last_precess = [                             #BN,sigmoid
            nn.BatchNorm2d(11),
            nn.Sigmoid()]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*last_precess)
        self.conv_end = nn.Conv2d(128, 11, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn_end = nn.BatchNorm2d(11)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    #
    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x) # input[2, 128, 56, 56]
        #print("res1_shape__",res1.shape) #-1,128,56,56
        res1_ =self.avg_res1(res1)
        res1_conv =self.conv_end(res1_)
        res2 = self.g2(res1) #-1,11,56,56
        res2_conv =self.conv_end(res2)  #
        res3 = self.g3(res2)
        res3_conv =self.conv_end(res3)
        w =  torch.add(torch.add(res1_,res2),res3)
        w =self.conv_end(w)
        out = w * res1_conv + w* res2_conv + w* res3_conv

        out = self.palayer(out)
        x = self.post(out)
        x = x.permute(0, 2, 3, 1)  # (-1,14,14,11)
        print(x.shape)  # 2,11,14,14
        return x

class Block_res(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=False,stride_change=False):
        super(Block_res, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        if stride_change==True:
            strides=1
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            # nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)

if __name__ == "__main__":
    import torch
    from torchsummary import summary
    net = FFA(gps=3, blocks=2)
    print(net)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    model = net.to(device)
    print(model)
    summary(model, (3, 448, 448))
