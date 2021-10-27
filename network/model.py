import torch.nn as nn
import torch.nn.functional as F
from utils import weights_init_normal
import torch

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator_H2F(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_H2F, self).__init__()

        # Initial convolution block
        self.conv1_L = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(2, 32, 7),
                                     nn.InstanceNorm2d(32),
                                     nn.ReLU(inplace=True))
        # 下采样
        self.downconv2_L = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.downconv3_L = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        # 残差网络 9层
        self.conv4_L = nn.Sequential(ResidualBlock(128))
        self.conv5_L = nn.Sequential(ResidualBlock(128))
        self.conv6_L = nn.Sequential(ResidualBlock(128))
        self.conv7_L = nn.Sequential(ResidualBlock(128))
        self.conv8_L = nn.Sequential(ResidualBlock(128))
        self.conv9_L = nn.Sequential(ResidualBlock(128))
        self.conv10_L = nn.Sequential(ResidualBlock(128))
        self.conv11_L = nn.Sequential(ResidualBlock(128))
        self.conv12_L = nn.Sequential(ResidualBlock(128))

        # 采样
        self.upconv13_L = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.upconv14_L = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(32),
                                        nn.ReLU(inplace=True))
        self.conv15_L = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, 1, 7))

        for p in self.parameters():
            p.requires_grad = False

        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(4, 32, 7),
                                     nn.InstanceNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv5_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv6_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv7_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv8_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv9_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv10_b = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128))
        self.conv11_b = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128))
        self.conv12_b = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(32),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    # 利用L通道得到的网络权值初始化H2F网络的权值
    def from_file(file_path: str) -> nn.Module:
        print('Initializes weights of H2F...')
        model = Generator_H2F(init_weights=True)

        temp = model.state_dict()

        print('Loading pretrained L weights...')
        state_dict = torch.load(file_path)

        pd = {}
        pd['conv1_L.1.weight'] = state_dict['model.1.weight']
        pd['conv1_L.1.bias'] = state_dict['model.1.bias']
        pd['downconv2_L.0.weight'] = state_dict['model.4.weight']
        pd['downconv2_L.0.bias'] = state_dict['model.4.bias']
        pd['downconv3_L.0.weight'] = state_dict['model.7.weight']
        pd['downconv3_L.0.bias'] = state_dict['model.7.bias']
        pd['conv4_L.0.conv_block.1.weight'] = state_dict['model.10.conv_block.1.weight']
        pd['conv4_L.0.conv_block.1.bias'] = state_dict['model.10.conv_block.1.bias']
        pd['conv4_L.0.conv_block.5.weight'] = state_dict['model.10.conv_block.5.weight']
        pd['conv4_L.0.conv_block.5.bias'] = state_dict['model.10.conv_block.5.bias']

        pd['conv5_L.0.conv_block.1.weight'] = state_dict['model.11.conv_block.1.weight']
        pd['conv5_L.0.conv_block.1.bias'] = state_dict['model.11.conv_block.1.bias']
        pd['conv5_L.0.conv_block.5.weight'] = state_dict['model.11.conv_block.5.weight']
        pd['conv5_L.0.conv_block.5.bias'] = state_dict['model.11.conv_block.5.bias']

        pd['conv6_L.0.conv_block.1.weight'] = state_dict['model.12.conv_block.1.weight']
        pd['conv6_L.0.conv_block.1.bias'] = state_dict['model.12.conv_block.1.bias']
        pd['conv6_L.0.conv_block.5.weight'] = state_dict['model.12.conv_block.5.weight']
        pd['conv6_L.0.conv_block.5.bias'] = state_dict['model.12.conv_block.5.bias']

        pd['conv7_L.0.conv_block.1.weight'] = state_dict['model.13.conv_block.1.weight']
        pd['conv7_L.0.conv_block.1.bias'] = state_dict['model.13.conv_block.1.bias']
        pd['conv7_L.0.conv_block.5.weight'] = state_dict['model.13.conv_block.5.weight']
        pd['conv7_L.0.conv_block.5.bias'] = state_dict['model.13.conv_block.5.bias']

        pd['conv8_L.0.conv_block.1.weight'] = state_dict['model.14.conv_block.1.weight']
        pd['conv8_L.0.conv_block.1.bias'] = state_dict['model.14.conv_block.1.bias']
        pd['conv8_L.0.conv_block.5.weight'] = state_dict['model.14.conv_block.5.weight']
        pd['conv8_L.0.conv_block.5.bias'] = state_dict['model.14.conv_block.5.bias']

        pd['conv9_L.0.conv_block.1.weight'] = state_dict['model.15.conv_block.1.weight']
        pd['conv9_L.0.conv_block.1.bias'] = state_dict['model.15.conv_block.1.bias']
        pd['conv9_L.0.conv_block.5.weight'] = state_dict['model.15.conv_block.5.weight']
        pd['conv9_L.0.conv_block.5.bias'] = state_dict['model.15.conv_block.5.bias']

        pd['conv10_L.0.conv_block.1.weight'] = state_dict['model.16.conv_block.1.weight']
        pd['conv10_L.0.conv_block.1.bias'] = state_dict['model.16.conv_block.1.bias']
        pd['conv10_L.0.conv_block.5.weight'] = state_dict['model.16.conv_block.5.weight']
        pd['conv10_L.0.conv_block.5.bias'] = state_dict['model.16.conv_block.5.bias']

        pd['conv11_L.0.conv_block.1.weight'] = state_dict['model.17.conv_block.1.weight']
        pd['conv11_L.0.conv_block.1.bias'] = state_dict['model.17.conv_block.1.bias']
        pd['conv11_L.0.conv_block.5.weight'] = state_dict['model.17.conv_block.5.weight']
        pd['conv11_L.0.conv_block.5.bias'] = state_dict['model.17.conv_block.5.bias']

        pd['conv12_L.0.conv_block.1.weight'] = state_dict['model.18.conv_block.1.weight']
        pd['conv12_L.0.conv_block.1.bias'] = state_dict['model.18.conv_block.1.bias']
        pd['conv12_L.0.conv_block.5.weight'] = state_dict['model.18.conv_block.5.weight']
        pd['conv12_L.0.conv_block.5.bias'] = state_dict['model.18.conv_block.5.bias']

        pd['upconv13_L.0.weight'] = state_dict['model.19.weight']
        pd['upconv13_L.0.bias'] = state_dict['model.19.bias']
        pd['upconv14_L.0.weight'] = state_dict['model.22.weight']
        pd['upconv14_L.0.bias'] = state_dict['model.22.bias']
        pd['conv15_L.1.weight'] = state_dict['model.26.weight']
        pd['conv15_L.1.bias'] = state_dict['model.26.bias']

        pd['downconv2_b.0.weight'] = state_dict['model.4.weight']
        pd['downconv2_b.0.bias'] = state_dict['model.4.bias']
        pd['downconv3_b.0.weight'] = state_dict['model.7.weight']
        pd['downconv3_b.0.bias'] = state_dict['model.7.bias']

        pd['conv4_b.1.weight'] = state_dict['model.10.conv_block.1.weight']
        pd['conv4_b.1.bias'] = state_dict['model.10.conv_block.1.bias']
        pd['conv4_b.5.weight'] = state_dict['model.10.conv_block.5.weight']
        pd['conv4_b.5.bias'] = state_dict['model.10.conv_block.5.bias']

        pd['conv5_b.1.weight'] = state_dict['model.11.conv_block.1.weight']
        pd['conv5_b.1.bias'] = state_dict['model.11.conv_block.1.bias']
        pd['conv5_b.5.weight'] = state_dict['model.11.conv_block.5.weight']
        pd['conv5_b.5.bias'] = state_dict['model.11.conv_block.5.bias']

        pd['conv6_b.1.weight'] = state_dict['model.12.conv_block.1.weight']
        pd['conv6_b.1.bias'] = state_dict['model.12.conv_block.1.bias']
        pd['conv6_b.5.weight'] = state_dict['model.12.conv_block.5.weight']
        pd['conv6_b.5.bias'] = state_dict['model.12.conv_block.5.bias']

        pd['conv7_b.1.weight'] = state_dict['model.13.conv_block.1.weight']
        pd['conv7_b.1.bias'] = state_dict['model.13.conv_block.1.bias']
        pd['conv7_b.5.weight'] = state_dict['model.13.conv_block.5.weight']
        pd['conv7_b.5.bias'] = state_dict['model.13.conv_block.5.bias']

        pd['conv8_b.1.weight'] = state_dict['model.14.conv_block.1.weight']
        pd['conv8_b.1.bias'] = state_dict['model.14.conv_block.1.bias']
        pd['conv8_b.5.weight'] = state_dict['model.14.conv_block.5.weight']
        pd['conv8_b.5.bias'] = state_dict['model.14.conv_block.5.bias']

        pd['conv9_b.1.weight'] = state_dict['model.15.conv_block.1.weight']
        pd['conv9_b.1.bias'] = state_dict['model.15.conv_block.1.bias']
        pd['conv9_b.5.weight'] = state_dict['model.15.conv_block.5.weight']
        pd['conv9_b.5.bias'] = state_dict['model.15.conv_block.5.bias']

        pd['conv10_b.1.weight'] = state_dict['model.16.conv_block.1.weight']
        pd['conv10_b.1.bias'] = state_dict['model.16.conv_block.1.bias']
        pd['conv10_b.5.weight'] = state_dict['model.16.conv_block.5.weight']
        pd['conv10_b.5.bias'] = state_dict['model.16.conv_block.5.bias']

        pd['conv11_b.1.weight'] = state_dict['model.17.conv_block.1.weight']
        pd['conv11_b.1.bias'] = state_dict['model.17.conv_block.1.bias']
        pd['conv11_b.5.weight'] = state_dict['model.17.conv_block.5.weight']
        pd['conv11_b.5.bias'] = state_dict['model.17.conv_block.5.bias']

        pd['conv12_b.1.weight'] = state_dict['model.18.conv_block.1.weight']
        pd['conv12_b.1.bias'] = state_dict['model.18.conv_block.1.bias']
        pd['conv12_b.5.weight'] = state_dict['model.18.conv_block.5.weight']
        pd['conv12_b.5.bias'] = state_dict['model.18.conv_block.5.bias']

        pd['upconv13_b.0.weight'] = state_dict['model.19.weight']
        pd['upconv13_b.0.bias'] = state_dict['model.19.bias']
        pd['upconv14_b.0.weight'] = state_dict['model.22.weight']
        pd['upconv14_b.0.bias'] = state_dict['model.22.bias']

        temp.update(pd)
        model.load_state_dict(temp)
        return model

    def forward(self, xin, xinl, mask, maskl):
        x_L = torch.cat((xinl, maskl), 1)
        x1_L = self.conv1_L(x_L)
        x2_L = self.downconv2_L(x1_L)
        x3_L = self.downconv3_L(x2_L)
        x4_L = self.conv4_L(x3_L)
        x5_L = self.conv5_L(x4_L)
        x6_L = self.conv6_L(x5_L)
        x7_L = self.conv7_L(x6_L)
        x8_L = self.conv8_L(x7_L)
        x9_L = self.conv9_L(x8_L)
        x10_L = self.conv10_L(x9_L)
        x11_L = self.conv11_L(x10_L)
        x12_L = self.conv12_L(x11_L)
        x_L = self.upconv13_L(x12_L)
        x_L = self.upconv14_L(x_L)
        x_L = self.conv15_L(x_L)
        xout_L = x_L + xinl

        x = torch.cat((xin, mask), 1)
        x1 = self.conv1_b(x)
        x2 = self.downconv2_b(x1)
        x3 = self.downconv3_b(x2)

        x4 = self.conv4_b(torch.mul(x3, x3_L)) + x3
        x5 = self.conv5_b(torch.mul(x4, x4_L)) + x4
        x6 = self.conv6_b(torch.mul(x5, x5_L)) + x5
        x7 = self.conv7_b(x6) + x6
        x8 = self.conv8_b(x7) + x7
        x9 = self.conv9_b(x8) + x8
        x10 = self.conv10_b(x9) + x9
        x11 = self.conv11_b(x10) + x10
        x12 = self.conv12_b(x11) + x11

        x = self.upconv13_b(x12)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        # return xout.tanh(),xout_L.tanh(),x3_L,x3,torch.mul(x3,x3_L),x4_L,x4,torch.mul(x4,x4_L),x5_L,x5,torch.mul(x5,x5_L)
        return xout.tanh(), xout_L.tanh()


class Generator_F2H(nn.Module):
    def __init__(self, init_weights=False):
        super(Generator_F2H, self).__init__()

        # Initial convolution block
        self.conv1_L = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(2, 32, 7),
                                     nn.InstanceNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.downconv2_L = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.downconv3_L = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_L = nn.Sequential(ResidualBlock(128))
        self.conv5_L = nn.Sequential(ResidualBlock(128))
        self.conv6_L = nn.Sequential(ResidualBlock(128))
        self.conv7_L = nn.Sequential(ResidualBlock(128))
        self.conv8_L = nn.Sequential(ResidualBlock(128))
        self.conv9_L = nn.Sequential(ResidualBlock(128))
        self.conv10_L = nn.Sequential(ResidualBlock(128))
        self.conv11_L = nn.Sequential(ResidualBlock(128))
        self.conv12_L = nn.Sequential(ResidualBlock(128))
        self.upconv13_L = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.upconv14_L = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(32),
                                        nn.ReLU(inplace=True))
        self.conv15_L = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, 1, 7))
        for p in self.parameters():
            p.requires_grad = False
        self.conv1_b = nn.Sequential(nn.ReflectionPad2d(3),
                                     nn.Conv2d(4, 32, 7),
                                     nn.InstanceNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.downconv2_b = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.downconv3_b = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(inplace=True))
        self.conv4_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv5_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv6_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv7_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv8_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv9_b = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.ReflectionPad2d(1),
                                     nn.Conv2d(128, 128, 3),
                                     nn.InstanceNorm2d(128))
        self.conv10_b = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128))
        self.conv11_b = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128))
        self.conv12_b = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(128, 128, 3),
                                      nn.InstanceNorm2d(128))
        self.upconv13_b = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(64),
                                        nn.ReLU(inplace=True))
        self.upconv14_b = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                                        nn.InstanceNorm2d(32),
                                        nn.ReLU(inplace=True))
        self.conv15_b = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(32, 3, 7))

        if init_weights:
            self.apply(weights_init_normal)

    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        print('Initializes weights of H2F...')
        model = Generator_F2H(init_weights=True)
        temp = model.state_dict()

        print('Loading pretrained L weights...')
        state_dict = torch.load(file_path)

        pd = {}

        # 对应 Conv2d()
        pd['conv1_L.1.weight'] = state_dict['model.1.weight']
        pd['conv1_L.1.bias'] = state_dict['model.1.bias']

        # downscaling 中网络由三层 Conv2d()在第一层
        pd['downconv2_L.0.weight'] = state_dict['model.4.weight']
        pd['downconv2_L.0.bias'] = state_dict['model.4.bias']
        pd['downconv3_L.0.weight'] = state_dict['model.7.weight']
        pd['downconv3_L.0.bias'] = state_dict['model.7.bias']

        # 对应ResidualBlock()  网络Conv2d在第2层和第6层
        pd['conv4_L.0.conv_block.1.weight'] = state_dict['model.10.conv_block.1.weight']
        pd['conv4_L.0.conv_block.1.bias'] = state_dict['model.10.conv_block.1.bias']
        pd['conv4_L.0.conv_block.5.weight'] = state_dict['model.10.conv_block.5.weight']
        pd['conv4_L.0.conv_block.5.bias'] = state_dict['model.10.conv_block.5.bias']

        pd['conv5_L.0.conv_block.1.weight'] = state_dict['model.11.conv_block.1.weight']
        pd['conv5_L.0.conv_block.1.bias'] = state_dict['model.11.conv_block.1.bias']
        pd['conv5_L.0.conv_block.5.weight'] = state_dict['model.11.conv_block.5.weight']
        pd['conv5_L.0.conv_block.5.bias'] = state_dict['model.11.conv_block.5.bias']

        pd['conv6_L.0.conv_block.1.weight'] = state_dict['model.12.conv_block.1.weight']
        pd['conv6_L.0.conv_block.1.bias'] = state_dict['model.12.conv_block.1.bias']
        pd['conv6_L.0.conv_block.5.weight'] = state_dict['model.12.conv_block.5.weight']
        pd['conv6_L.0.conv_block.5.bias'] = state_dict['model.12.conv_block.5.bias']

        pd['conv7_L.0.conv_block.1.weight'] = state_dict['model.13.conv_block.1.weight']
        pd['conv7_L.0.conv_block.1.bias'] = state_dict['model.13.conv_block.1.bias']
        pd['conv7_L.0.conv_block.5.weight'] = state_dict['model.13.conv_block.5.weight']
        pd['conv7_L.0.conv_block.5.bias'] = state_dict['model.13.conv_block.5.bias']

        pd['conv8_L.0.conv_block.1.weight'] = state_dict['model.14.conv_block.1.weight']
        pd['conv8_L.0.conv_block.1.bias'] = state_dict['model.14.conv_block.1.bias']
        pd['conv8_L.0.conv_block.5.weight'] = state_dict['model.14.conv_block.5.weight']
        pd['conv8_L.0.conv_block.5.bias'] = state_dict['model.14.conv_block.5.bias']

        pd['conv9_L.0.conv_block.1.weight'] = state_dict['model.15.conv_block.1.weight']
        pd['conv9_L.0.conv_block.1.bias'] = state_dict['model.15.conv_block.1.bias']
        pd['conv9_L.0.conv_block.5.weight'] = state_dict['model.15.conv_block.5.weight']
        pd['conv9_L.0.conv_block.5.bias'] = state_dict['model.15.conv_block.5.bias']

        pd['conv10_L.0.conv_block.1.weight'] = state_dict['model.16.conv_block.1.weight']
        pd['conv10_L.0.conv_block.1.bias'] = state_dict['model.16.conv_block.1.bias']
        pd['conv10_L.0.conv_block.5.weight'] = state_dict['model.16.conv_block.5.weight']
        pd['conv10_L.0.conv_block.5.bias'] = state_dict['model.16.conv_block.5.bias']

        pd['conv11_L.0.conv_block.1.weight'] = state_dict['model.17.conv_block.1.weight']
        pd['conv11_L.0.conv_block.1.bias'] = state_dict['model.17.conv_block.1.bias']
        pd['conv11_L.0.conv_block.5.weight'] = state_dict['model.17.conv_block.5.weight']
        pd['conv11_L.0.conv_block.5.bias'] = state_dict['model.17.conv_block.5.bias']

        pd['conv12_L.0.conv_block.1.weight'] = state_dict['model.18.conv_block.1.weight']
        pd['conv12_L.0.conv_block.1.bias'] = state_dict['model.18.conv_block.1.bias']
        pd['conv12_L.0.conv_block.5.weight'] = state_dict['model.18.conv_block.5.weight']
        pd['conv12_L.0.conv_block.5.bias'] = state_dict['model.18.conv_block.5.bias']

        # 采样层 中网络由三层组成 ConvTranspose2d()在第1层
        pd['upconv13_L.0.weight'] = state_dict['model.19.weight']
        pd['upconv13_L.0.bias'] = state_dict['model.19.bias']
        pd['upconv14_L.0.weight'] = state_dict['model.22.weight']
        pd['upconv14_L.0.bias'] = state_dict['model.22.bias']

        pd['conv15_L.1.weight'] = state_dict['model.26.weight']
        pd['conv15_L.1.bias'] = state_dict['model.26.bias']

        pd['downconv2_b.0.weight'] = state_dict['model.4.weight']
        pd['downconv2_b.0.bias'] = state_dict['model.4.bias']
        pd['downconv3_b.0.weight'] = state_dict['model.7.weight']
        pd['downconv3_b.0.bias'] = state_dict['model.7.bias']

        pd['conv4_b.1.weight'] = state_dict['model.10.conv_block.1.weight']
        pd['conv4_b.1.bias'] = state_dict['model.10.conv_block.1.bias']
        pd['conv4_b.5.weight'] = state_dict['model.10.conv_block.5.weight']
        pd['conv4_b.5.bias'] = state_dict['model.10.conv_block.5.bias']

        pd['conv5_b.1.weight'] = state_dict['model.11.conv_block.1.weight']
        pd['conv5_b.1.bias'] = state_dict['model.11.conv_block.1.bias']
        pd['conv5_b.5.weight'] = state_dict['model.11.conv_block.5.weight']
        pd['conv5_b.5.bias'] = state_dict['model.11.conv_block.5.bias']

        pd['conv6_b.1.weight'] = state_dict['model.12.conv_block.1.weight']
        pd['conv6_b.1.bias'] = state_dict['model.12.conv_block.1.bias']
        pd['conv6_b.5.weight'] = state_dict['model.12.conv_block.5.weight']
        pd['conv6_b.5.bias'] = state_dict['model.12.conv_block.5.bias']

        pd['conv7_b.1.weight'] = state_dict['model.13.conv_block.1.weight']
        pd['conv7_b.1.bias'] = state_dict['model.13.conv_block.1.bias']
        pd['conv7_b.5.weight'] = state_dict['model.13.conv_block.5.weight']
        pd['conv7_b.5.bias'] = state_dict['model.13.conv_block.5.bias']

        pd['conv8_b.1.weight'] = state_dict['model.14.conv_block.1.weight']
        pd['conv8_b.1.bias'] = state_dict['model.14.conv_block.1.bias']
        pd['conv8_b.5.weight'] = state_dict['model.14.conv_block.5.weight']
        pd['conv8_b.5.bias'] = state_dict['model.14.conv_block.5.bias']

        pd['conv9_b.1.weight'] = state_dict['model.15.conv_block.1.weight']
        pd['conv9_b.1.bias'] = state_dict['model.15.conv_block.1.bias']
        pd['conv9_b.5.weight'] = state_dict['model.15.conv_block.5.weight']
        pd['conv9_b.5.bias'] = state_dict['model.15.conv_block.5.bias']

        pd['conv10_b.1.weight'] = state_dict['model.16.conv_block.1.weight']
        pd['conv10_b.1.bias'] = state_dict['model.16.conv_block.1.bias']
        pd['conv10_b.5.weight'] = state_dict['model.16.conv_block.5.weight']
        pd['conv10_b.5.bias'] = state_dict['model.16.conv_block.5.bias']

        pd['conv11_b.1.weight'] = state_dict['model.17.conv_block.1.weight']
        pd['conv11_b.1.bias'] = state_dict['model.17.conv_block.1.bias']
        pd['conv11_b.5.weight'] = state_dict['model.17.conv_block.5.weight']
        pd['conv11_b.5.bias'] = state_dict['model.17.conv_block.5.bias']

        pd['conv12_b.1.weight'] = state_dict['model.18.conv_block.1.weight']
        pd['conv12_b.1.bias'] = state_dict['model.18.conv_block.1.bias']
        pd['conv12_b.5.weight'] = state_dict['model.18.conv_block.5.weight']
        pd['conv12_b.5.bias'] = state_dict['model.18.conv_block.5.bias']

        pd['upconv13_b.0.weight'] = state_dict['model.19.weight']
        pd['upconv13_b.0.bias'] = state_dict['model.19.bias']
        pd['upconv14_b.0.weight'] = state_dict['model.22.weight']
        pd['upconv14_b.0.bias'] = state_dict['model.22.bias']

        temp.update(pd)
        model.load_state_dict(temp)
        return model

    # 在残差网络部分使用乘法连接
    def forward(self, xin, xinl, mask, maskl):
        x_L = torch.cat((xinl, maskl), 1)
        x1_L = self.conv1_L(x_L)
        x2_L = self.downconv2_L(x1_L)
        x3_L = self.downconv3_L(x2_L)
        x4_L = self.conv4_L(x3_L)
        x5_L = self.conv5_L(x4_L)
        x6_L = self.conv6_L(x5_L)
        x7_L = self.conv7_L(x6_L)
        x8_L = self.conv8_L(x7_L)
        x9_L = self.conv9_L(x8_L)
        x10_L = self.conv10_L(x9_L)
        x11_L = self.conv11_L(x10_L)
        x12_L = self.conv12_L(x11_L)
        x_L = self.upconv13_L(x12_L)
        x_L = self.upconv14_L(x_L)
        x_L = self.conv15_L(x_L)
        xout_L = x_L + xinl

        x = torch.cat((xin, mask), 1)
        x1 = self.conv1_b(x)
        x2 = self.downconv2_b(x1)
        x3 = self.downconv3_b(x2)
        x4 = self.conv4_b(torch.mul(x3, x3_L)) + x3
        x5 = self.conv5_b(torch.mul(x4, x4_L)) + x4
        x6 = self.conv6_b(torch.mul(x5, x5_L)) + x5
        x7 = self.conv7_b(x6) + x6
        x8 = self.conv8_b(x7) + x7
        x9 = self.conv9_b(x8) + x8
        x10 = self.conv10_b(x9) + x9
        x11 = self.conv11_b(x10) + x10
        x12 = self.conv12_b(x11) + x11

        x = self.upconv13_b(x12)
        x = self.upconv14_b(x)
        x = self.conv15_b(x)
        xout = x + xin
        return xout.tanh(), xout_L.tanh()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(3, 32, 4, stride=2, padding=1),
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
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1).squeeze(0)  # global avg pool
