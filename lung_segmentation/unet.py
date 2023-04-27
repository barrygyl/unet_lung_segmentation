
"""
这是根据UNet模型搭建出的一个基本网络结构
输入和输出大小是一样的，可以根据需求进行修改
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块。3x3 ReLU，卷积具有输入输出通道，每次不一样，第一次是64 64，第二次是128 128，第三次是256 256。卷积两次
class Conv(nn.Module):  # 继承nn.Module
    def __init__(self, C_in, C_out):  # 输入输出通道，每次不一样，C_in, C_out定义不能固定死
        super(Conv, self).__init__()  # 继承父类方法
        # self.layer用序列构造器nn.Sequential()做
        self.layer = nn.Sequential(

            # nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            # 3x3的卷积，步长为1，padding为1，reflect反射可以加强特征提取的能力,bias=False的原因是要使用nn.BatchNorm2d()
            # 3x3的卷积，步长为1，padding为1，
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),  # nn.Dropout2d(0.3),
            nn.LeakyReLU(),  # 因为要激活ReLU()

            # 第二个卷积同理
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),  # nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):  # 前向运算
        return self.layer(x)


# 下采样模块，采用3x3卷积，步长为2.因为最大池化max pool 2x2 没有特征提取的能力，丢特征丢的比较多
class DownSampling(nn.Module):  # 继承nn.Module
    def __init__(self, channel):  # 下采样也有数据通道C
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            # nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.Conv2d(channel, channel, 3, 2, 1),

            # nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
# 上采样后得到的特征图，和之前的特征图copy拼接后在进行卷积，再上采样，拼接，卷积
# 上采样up-conv 2x2。有转置卷积（周围填充一层空洞卷积，让图像变大）和插值法。本次用最临近插值法（直接导入），空洞卷积的空洞对图像分割影响有点大。
class UpSampling(nn.Module):

    def __init__(self, channel):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        # 使用1X1的卷积进行降通道,不进行像素特征提取，channel//2:变为原来的一半(1024->512,512->256)
        self.Up = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        # 使用邻近插值进行下采样
        # scale_factor=2表示变为原来的两倍，mode='nearest'类型为最临近
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 使用torch.cat()进行拼接，当前上采样的，和之前下采样过程中的。
        return torch.cat((x, feature_map), dim=1)  # 结构为NCHW,在C通道进行的为0123。N是batch大小


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        self.C1 = Conv(3, 64)  # input image.卷积。输入是3，卷成64
        self.D1 = DownSampling(64)  # 下采样
        self.C2 = Conv(64, 128)  # 64卷为128
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)  # 为了降通道，前向待会再算
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()  # 激活。虽然为彩色图像，但是这是一个二分类，只用区分0无颜色，1有颜色，所以采用Sigmoid()
        self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)  # 因为要输出彩色图片，为三通道.3x3卷积，步长为1，padding为1

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)  # 对x进行卷积
        R2 = self.C2(self.D1(R1))  # 对卷积后的R1进行下采样，对下采样后的图像进行c2卷积，存为R2
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))  # 对Y1进行上采样，R5与R4进行拼接后再卷积
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))


if __name__ == '__main__':
    a = torch.randn(2, 3, 256, 256)  # 2 批次？3 通道？256x256.输出为相同的则没有问题
    net = UNet()
    print(net(a).shape)
    print(net)

