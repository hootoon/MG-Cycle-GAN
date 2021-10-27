import torch.nn as nn
import torch.nn.functional as F
import torch

# 残差网络  解决梯度消失问题
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        # super(ResidualBlock, self) 首先找到 ResidualBlock的父类（就是类 nn.Module），然后把类 ResidualBlock的对象转换为类 nn.Module 的对象
        super(ResidualBlock, self).__init__()

        # nn.ReflectionPad2d 填充边界 使用镜像填充的方法，以边界的一行或一列为对称轴
        conv_block = [nn.ReflectionPad2d(1),
                      # 二维卷积处理二维数据， 输入通道、输出通道、卷积核大小
                      nn.Conv2d(in_features, in_features, 3),
                      # 归一化操作，用在图像像素上，应用于RGB图像等信道数据的每一个信道
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        # nn.Sequential一个有序的神经网络容器，依次将网络模块添加到其中
        self.conv_block = nn.Sequential(*conv_block)

    # forward函数 通过for循环依次调用添加到self._module中的子模块，最后输出经过所有神经网络层的结果
    def forward(self, x):
        return x + self.conv_block(x)


class Generator_H2F(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_H2F, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc+1, 32, 7),
                 nn.InstanceNorm2d(32),
                 # 激活函数ReLU
                 nn.ReLU(inplace=True)]

        # Downscaling
        in_features = 32
        out_features = in_features * 2
        for _ in range(2):
            # stride 卷积的步幅，padding 填充
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Sampling 采样
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(32, output_nc, 7)]
        # nn.Tanh() ]
        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        # 激活函数 tanh()
        return (self.model(torch.cat((x, mask), 1)) + x).tanh()
        # return (self.model(x) + x).tanh()  # (min=-1, max=1) #just learn a residual


class Generator_F2H(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator_F2H, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc + 1, 32, 7),  # + mask 加上掩膜层
                 nn.InstanceNorm2d(32),
                 nn.ReLU(inplace=True)]

        # Downscaling
        in_features = 32
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Sampling
        out_features = in_features // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(32, output_nc, 7)]
        # nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        # torch.cat将两个张量（tensor）拼接在一起  0时按行进行拼接，1时按列进行拼接
        return (self.model(torch.cat((x, mask), 1)) + x).tanh()  # (min=-1, max=1) #just learn a residual


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 32, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(32, 64, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        model += [nn.Conv2d(256, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        # .view(x.size()[0], -1) 将前面操作输出的多维度的tensor展平成一维，然后输入分类器，-1是自适应分配，指在不知道函数有多少列的情况下，根据原tensor数据自动分配列数
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).view(1)  # global avg pool
