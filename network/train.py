from __future__ import print_function
import os
import argparse
import itertools
import sys

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from utils import ReplayBuffer
from utils import LambdaLR
from utils import mask_generator,mask_generator_lab
from utils import QueueMask_all
from model import Generator_F2H,Generator_H2F,Discriminator
from datasets import ImageDataset
import numpy as np
from utils import weights_init_normal
print(torch.cuda.is_available())
os.environ["CUDA_VISIBLE_DEVICES"]="4"
from skimage import io, color
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore")

###################################

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
    opt = parser.parse_args()

    opt.dataroot = 'SHIQ'

    print(torch.cuda.device_count())
    if not os.path.exists('model'):
        os.mkdir('model')
    opt.log_path = os.path.join('model', str('log') + '.txt')

    if torch.cuda.is_available():
        opt.cuda = True

    print(opt)

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator_H2F.from_file('model/netG_A2B.pth')  # highlight to highlight_free
    netG_B2A = Generator_F2H.from_file('model/netG_B2A.pth')  # highlight_free to highlight
    netD_A = Discriminator()
    netD_B = Discriminator()

    netG_A2B.cuda()
    netG_B2A.cuda()
    netD_A.cuda()
    netD_B.cuda()

    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()  # lsgan
    # criterion_GAN = torch.nn.BCEWithLogitsLoss() #vanilla
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers

    # optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=opt.lr, betas=(0.5, 0.999))
    optimizer_G = torch.optim.Adam(
        filter(lambda p: p.requires_grad, itertools.chain(netG_A2B.parameters(), netG_B2A.parameters())), lr=opt.lr,
        betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                       opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                                           opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, 3, opt.size, opt.size)
    input_A_L = Tensor(opt.batchSize, 1, opt.size, opt.size)
    input_B_L = Tensor(opt.batchSize, 1, opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
    mask_non_highlight = Variable(Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(-1.0),
                               requires_grad=False)  # -1.0 non-highlight
    angle_real = Variable(Tensor(opt.batchSize, opt.size, opt.size).fill_(1.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True),
                            batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

    curr_iter = 0
    G_losses_temp = 0
    D_A_losses_temp = 0
    D_B_losses_temp = 0
    G_losses = []
    D_A_losses = []
    D_B_losses = []

    mask_queue = QueueMask_all(dataloader.__len__() / 4)
    open(opt.log_path, 'w').write(str(opt) + '\n\n')

    def colour_angle(ff, rr):
        return torch.mul(ff, rr).sum(1) / (torch.mul(torch.pow((torch.pow(ff, 2)).sum(1), 0.5),
                                                     torch.pow((torch.pow(rr, 2)).sum(1), 0.5)) + 1e-8)

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A']))
            real_B = Variable(input_B.copy_(batch['B']))
            real_A_L = Variable(input_A_L.copy_(batch['AL']))
            real_B_L = Variable(input_B_L.copy_(batch['BL']))

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B, _ = netG_A2B(real_B, real_B_L)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0  # ||Gb(b)-b||1
            # G_B2A(A) should equal A if real A is fed, so the mask should be all zeros
            same_A, _ = netG_B2A(real_A, real_A_L, mask_non_highlight, mask_non_highlight)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0  # ||Ga(a)-a||1

            # GAN loss
            fake_B, fake_B_L = netG_A2B(real_A, real_A_L)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)  # log(Db(Gb(a)))

            mask_queue.insert(mask_generator(real_A, fake_B), mask_generator_lab(real_A_L, fake_B_L))
            mask, mask_L = mask_queue.rand_item()
            fake_A, fake_A_L = netG_B2A(real_B, real_B_L, mask, mask_L)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)  # log(Da(Ga(b)))

            # Cycle loss
            cmask, cmask_L = mask_queue.last_item()
            recovered_A, _ = netG_B2A(fake_B, fake_B_L, cmask, cmask_L)  # real highlight, false highlight free
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0  # ||Ga(Gb(a))-a||1

            recovered_B, _ = netG_A2B(fake_A, fake_A_L)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0  # ||Gb(Ga(b))-b||1

            # Colour loss
            loss_colour_A = criterion_cycle(colour_angle(recovered_A, real_A), angle_real) * 10.0
            loss_colour_B = criterion_cycle(colour_angle(recovered_B, real_B), angle_real) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_colour_A + loss_colour_B
            loss_G.backward()

            # G_losses.append(loss_G.item())
            G_losses_temp += loss_G.item()

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)  # log(Da(a))

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)  # log(1-Da(G(b)))

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            # D_A_losses.append(loss_D_A.item())
            D_A_losses_temp += loss_D_A.item()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)  # log(Db(b))

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)  # log(1-Db(G(a)))

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            # D_B_losses.append(loss_D_B.item())
            D_B_losses_temp += loss_D_B.item()

            optimizer_D_B.step()
            ###################################

            curr_iter += 1
            if (i + 1) % opt.iter_loss == 0:
                log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_G_identity %.5f], [loss_G_GAN %.5f],' \
                      '[loss_G_cycle %.5f], [loss_D %.5f], [loss_colour %.5f]' % \
                      (epoch, curr_iter, loss_G, (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A),
                       (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B), (loss_colour_A + loss_colour_B))
                print(log)
                open(opt.log_path, 'a').write(log + '\n')

                G_losses.append(G_losses_temp / opt.iter_loss)
                D_A_losses.append(D_A_losses_temp / opt.iter_loss)
                D_B_losses.append(D_B_losses_temp / opt.iter_loss)
                G_losses_temp = 0
                D_A_losses_temp = 0
                D_B_losses_temp = 0

                avg_log = '[the last %d iters], [loss_G %.5f], [D_A_losses %.5f], [D_B_losses %.5f],' \
                          % (opt.iter_loss, G_losses[G_losses.__len__() - 1], D_A_losses[D_A_losses.__len__() - 1], \
                             D_B_losses[D_B_losses.__len__() - 1])
                print(avg_log)
                open(opt.log_path, 'a').write(avg_log + '\n')

                outputhighlightfreeimage = fake_B.data
                outputhighlightfreeimage[:, 0] = 50.0 * (outputhighlightfreeimage[:, 0] + 1.0)
                outputhighlightfreeimage[:, 1:] = 255.0 * (outputhighlightfreeimage[:, 1:] + 1.0) / 2.0 - 128.0
                outputhighlightfreeimage = outputhighlightfreeimage.data.squeeze(0).cpu()
                outputhighlightfreeimage = outputhighlightfreeimage.transpose(0, 2).transpose(0, 1).contiguous().numpy()

                outputhighlightfreeimage = resize(outputhighlightfreeimage, (480, 640, 3))
                outputimagerealsr = color.lab2rgb(outputhighlightfreeimage)
                outputimagerealsr = (outputimagerealsr*255).astype(np.uint8)
                io.imsave('./model/outputhighlightfreeimage.png', outputimagerealsr)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'model/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'model/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'model/netD_A.pth')
        torch.save(netD_B.state_dict(), 'model/netD_B.pth')

        if (epoch + 1) > 90:
            torch.save(netG_A2B.state_dict(), ('model/netG_A2B_%d.pth' % (epoch + 1)))
            torch.save(netG_B2A.state_dict(), ('model/netG_B2A_%d.pth' % (epoch + 1)))
            torch.save(netD_A.state_dict(), ('model/netD_A_%d.pth' % (epoch + 1)))
            torch.save(netD_B.state_dict(), ('model/netD_B_%d.pth' % (epoch + 1)))

        print('Epoch:{}'.format(epoch))


if __name__ == '__main__':
    sys.exit(main())
