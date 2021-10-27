import random
import time
import datetime
import torch.nn as nn
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import threshold_otsu
from skimage import io, color

to_pil = transforms.ToPILImage()
# 三通道彩色图像转单通道灰度图像
to_gray = transforms.Grayscale(num_output_channels=1)

#阴影掩膜队列
class QueueMask_all():
    def __init__(self, length):
        self.max_length = length
        self.queue = []
        self.queue_L = []

    def insert(self, mask, mask_L):
        if self.queue.__len__() >= self.max_length:
            # pop(0) 弹出列表第一个元素
            self.queue.pop(0)
        if self.queue_L.__len__() >= self.max_length:
            self.queue_L.pop(0)

        self.queue.append(mask)
        self.queue_L.append(mask_L)

    # 取列表中的随机掩膜
    def rand_item(self):
        # assert断言，当后续表达式为假时，触发异常
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        assert self.queue_L.__len__() > 0, 'Error! Empty queue!'
        temp = np.random.randint(0, self.queue.__len__())
        return self.queue[temp], self.queue_L[temp]

    # 取列表中的最后一个掩膜
    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        assert self.queue_L.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__() - 1], self.queue_L[self.queue.__len__() - 1]

# 阴影掩膜队列
class QueueMask():
    def __init__(self, length):
        self.max_length = length
        self.queue = []

    def insert(self, mask):
        if self.queue.__len__() >= self.max_length:
            self.queue.pop(0)

        self.queue.append(mask)

    def rand_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[np.random.randint(0, self.queue.__len__())]

    def last_item(self):
        assert self.queue.__len__() > 0, 'Error! Empty queue!'
        return self.queue[self.queue.__len__() - 1]


# 高光掩膜生成
def mask_generator(highlight, highlight_free):
    im_f = to_gray(to_pil(((highlight_free.data.squeeze(0) + 1.0) * 0.5).cpu()))
    im_s = to_gray(to_pil(((highlight.data.squeeze(0) + 1.0) * 0.5).cpu()))
    diff = (np.asarray(im_f, dtype='float32') - np.asarray(im_s,dtype='float32'))
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L) - 0.5) / 0.5).unsqueeze(0).unsqueeze(
        0).cuda()  # -1.0:non-highlight, 1.0:highlight
    mask.requires_grad = False

    return mask

# 亮度通道高光掩膜生成
def mask_generator_lab(highlight, highlight_free):
    # tensor格式数据转uint8图像
    im_f = to_pil(((highlight_free.data.squeeze(0) + 1.0) * 0.5).cpu())
    im_s = to_pil(((highlight.data.squeeze(0) + 1.0) * 0.5).cpu())

    diff = (np.asarray(im_f, dtype='float32') - np.asarray(im_s,
                                                           dtype='float32'))  # difference between highlight image and highlight_free image
    L = threshold_otsu(diff)
    mask = torch.tensor((np.float32(diff >= L) - 0.5) / 0.5).unsqueeze(0).unsqueeze(
        0).cuda()  # -1.0:non-highlight, 1.0:highlight
    mask.requires_grad = False

    return mask

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    # shape[0]图像的垂直尺寸（高度）
    if image.shape[0] == 1:
        # np.tile() 赋值数组 （3,1,1）将最低维度复制3倍
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

# 学习速率
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

# 初始化网络权值参数
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #  torch.nn.init.normal_(tensor,mean,std)高斯分布，从给定均值和标准差的正太分布（0,0.02）中生成值，填充m.weight.data
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        # torch.nn.init.constant_（tensor，val）将Tensor填充为常量值，这里用值0填充张量
        torch.nn.init.constant_(m.bias.data, 0.0)


class cyclemaskloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake_B, real_A, mask):
        mask = (1.0 - mask) / 2.0
        # 扩充张量mask 追加维度
        mask = mask.repeat(1, 3, 1, 1)
        mask.requires_grad = False
        return torch.mean(torch.pow((torch.mul(fake_B, mask) - torch.mul(real_A, mask)), 2))
