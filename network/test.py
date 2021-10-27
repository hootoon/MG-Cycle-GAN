import argparse
import os
from os.path import exists, join as join_paths
import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize

from model import Generator_H2F

print(torch.cuda.is_available())
os.environ["CUDA_VISIBLE_DEVICES"] = "4,2,6,1,7,5,3,0"

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--generator_A2B', type=str, default='model/netG_A2B.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()
with torch.no_grad():
    # 'SHIQ_data/shiq_test/A'
    opt.dataroot_A = 'SHIQ/test/test'
    opt.im_suf_A = '.png'

    opt.dataroot_M = 'SHIQ/test/t'
    opt.im_suf_M1 = 'A'
    opt.im_suf_M2 = 'T'

    if torch.cuda.is_available():
        opt.cuda = True
        device = torch.device('cuda:0')

    netG_A2B = Generator_H2F()

    if opt.cuda:
        netG_A2B.to(device)

    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_A2B.eval()

    if not os.path.exists('model/Result'):
        os.makedirs('model/Result')

    gt_list = [os.path.splitext(f)[0] for f in os.listdir(opt.dataroot_A) if f.endswith(opt.im_suf_A)]

    for idx, img_name in enumerate(gt_list):
        # Set model input
        labimage = color.rgb2lab(io.imread(os.path.join(opt.dataroot_A, img_name + opt.im_suf_A)))

        img_name = img_name[:6] + 'T'
        maskimage = io.imread(os.path.join(opt.dataroot_M, img_name + opt.im_suf_A))

        mask = resize(maskimage, (200, 200))

        for i in range(0, 200):
            for j in range(0, 200):
                if (mask[i, j] > 0):
                    mask[i, j] = 1
                else:
                    mask[i, j] = -1

        # 将item_A转化为张量
        mask = torch.from_numpy(mask).float()
        # 重构张量维度
        mask = mask.view(200, 200, 1)
        mask = mask.transpose(0, 1).transpose(0, 2).contiguous()
        mask_var = mask.unsqueeze(0).to(device)

        labimage448 = resize(labimage, (200, 200, 3))
        labimage_L448 = labimage448[:, :, 0]
        labimage448[:, :, 0] = np.asarray(labimage448[:, :, 0]) / 50.0 - 1.0
        labimage448[:, :, 1:] = 2.0 * (np.asarray(labimage448[:, :, 1:]) + 128.0) / 255.0 - 1.0
        labimage448 = torch.from_numpy(labimage448).float()
        labimage_L448 = labimage448[:, :, 0]
        labimage448 = labimage448.view(200, 200, 3)
        labimage_L448 = labimage_L448.view(200, 200, 1)
        labimage448 = labimage448.transpose(0, 1).transpose(0, 2).contiguous()
        labimage448 = labimage448.unsqueeze(0).to(device)
        labimage_L448 = labimage_L448.transpose(0, 1).transpose(0, 2).contiguous()
        labimage_L448 = labimage_L448.unsqueeze(0).to(device)

        labimage480 = resize(labimage, (200, 200, 3))
        labimage_L480 = labimage480[:, :, 0]
        labimage480[:, :, 0] = np.asarray(labimage480[:, :, 0]) / 50.0 - 1.0
        labimage480[:, :, 1:] = 2.0 * (np.asarray(labimage480[:, :, 1:]) + 128.0) / 255.0 - 1.0
        labimage480 = torch.from_numpy(labimage480).float()
        labimage_L480 = labimage480[:, :, 0]
        labimage480 = labimage480.view(200, 200, 3)
        labimage_L480 = labimage_L480.view(200, 200, 1)
        labimage480 = labimage480.transpose(0, 1).transpose(0, 2).contiguous()
        labimage480 = labimage480.unsqueeze(0).to(device)
        labimage_L480 = labimage_L480.transpose(0, 1).transpose(0, 2).contiguous()
        labimage_L480 = labimage_L480.unsqueeze(0).to(device)

        # Generate output
        temp_B448, _ = netG_A2B(labimage448, labimage_L448, mask_var, mask_var)
        temp_B480, _ = netG_A2B(labimage480, labimage_L480, mask_var, mask_var)

        fake_B448 = temp_B448.data
        # fake_B448[:,0]=50.0*(fake_B448[:,0]+1.0)
        fake_B448[:, 1:] = 255.0 * (fake_B448[:, 1:] + 1.0) / 2.0 - 128.0
        fake_B448 = fake_B448.data.squeeze(0).cpu()
        fake_B448 = fake_B448.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        fake_B448 = resize(fake_B448, (200, 200, 3))

        fake_B480 = temp_B480.data
        fake_B480[:, 0] = 50.0 * (fake_B480[:, 0] + 1.0)
        # fake_B480[:,1:]=255.0*(fake_B480[:,1:]+1.0)/2.0-128.0
        fake_B480 = fake_B480.data.squeeze(0).cpu()
        fake_B480 = fake_B480.transpose(0, 2).transpose(0, 1).contiguous().numpy()
        fake_B480 = resize(fake_B480, (200, 200, 3))

        fake_B = fake_B480
        fake_B[:, :, 1:] = fake_B448[:, :, 1:]
        outputimage = color.lab2rgb(fake_B)
        outputimage = (outputimage * 255).astype(np.uint8)
        save_result = join_paths('./model/Result/%s' % (img_name + opt.im_suf_A))
        io.imsave(save_result, outputimage)
        print('Generated images %04d of %04d' % (idx + 1, len(gt_list)))