from __future__ import print_function
import os
import datetime
import argparse
import itertools
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from utils import ReplayBuffer
from utils import LambdaLR
from utils import weights_init_normal
from utils import mask_generator_lab
from utils import QueueMask
from model_l import Generator_F2H, Generator_H2F, Discriminator
from datasets_l import ImageDataset
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io


def main(argv=None):
    print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # 用来解析命令行参数  首先声明一个parser
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=400, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_false', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--snapshot_epochs', type=int, default=2, help='number of epochs of training')
    parser.add_argument('--iter_loss', type=int, default=100, help='average loss for n iterations')
    # 读取命令行参数
    opt = parser.parse_args()

    opt.dataroot = 'SHIQ_data'

    if not os.path.exists('model_l'):
        os.mkdir('model_l')
    # os.path.join() 将路径名和文件合成一个路径
    opt.log_path = os.path.join('model_l', str('log') + '.txt')

    if torch.cuda.is_available():
        opt.cuda = True

    ###### Definition of variables ######
    # Networks
    netG_A2B = Generator_H2F(opt.input_nc, opt.output_nc)  # highlight to highlight_free
    netG_B2A = Generator_F2H(opt.output_nc, opt.input_nc)  # highlight_free to highlight
    netD_A = Discriminator(opt.input_nc)
    netD_B = Discriminator(opt.output_nc)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    # 使用apply初始化网络权值参数，apply函数会递归地搜索网络内所有module，并把参数表示的函数应用到所有module上
    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()  # lsgan
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                          opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                            opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
    input_C = Tensor(opt.batchSize, opt.output_nc,opt.size, opt.size)
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)
    mask_non_highlight = Variable(Tensor(opt.batchSize, 1, opt.size, opt.size).fill_(-1.0),
                               requires_grad=False)  # -1.0 non-highlight

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

    mask_queue = QueueMask(dataloader.__len__() / 4)
    open(opt.log_path, 'w').write(str(opt) + '\n\n')

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
                # Set model input
                real_A = Variable(input_A.copy_(batch['A']))
                real_B = Variable(input_B.copy_(batch['B']))
                mask_real = Variable(input_C.copy_(batch['C']))

                ###### Generators A2B and B2A ######
                optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = netG_A2B(real_B,mask_non_highlight)
                loss_identity_B = criterion_identity(same_B, real_B) * 5.0  # ||Gb(b)-b||1
                # G_B2A(A) should equal A if real A is fed, so the mask should be all zeros
                same_A = netG_B2A(real_A, mask_non_highlight)
                loss_identity_A = criterion_identity(same_A, real_A) * 5.0  # ||Ga(a)-a||1

                # GAN loss
                fake_B = netG_A2B(real_A,mask_real)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real)  # log(Db(Gb(a)))

                mask = mask_real
                mask_queue.insert(mask)

                mask_random = mask_queue.rand_item()
                fake_A = netG_B2A(real_B, mask_random)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real)  # log(Da(Ga(b)))

                # Cycle loss
                recovered_A = netG_B2A(fake_B, mask_queue.last_item())  # real highlight, false highlight free
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0  # ||Ga(Gb(a))-a||1

                recovered_B = netG_A2B(fake_A,mask_random)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0  # ||Gb(Ga(b))-b||1

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
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

                if (i + 1) % iter_loss == 0:
                    log = 'Epoch: %d, [iter %d], [loss_G %.5f], [loss_G_identity %.5f], [loss_G_GAN %.5f],' \
                          '[loss_G_cycle %.5f], [loss_D %.5f]' % \
                          (epoch, curr_iter, loss_G, (loss_identity_A + loss_identity_B), (loss_GAN_A2B + loss_GAN_B2A),
                           (loss_cycle_ABA + loss_cycle_BAB), (loss_D_A + loss_D_B))
                    print(log)
                    open(log_path, 'a').write(log + '\n')

                    G_losses.append(G_losses_temp / iter_loss)
                    D_A_losses.append(D_A_losses_temp / iter_loss)
                    D_B_losses.append(D_B_losses_temp / iter_loss)
                    G_losses_temp = 0
                    D_A_losses_temp = 0
                    D_B_losses_temp = 0

                    avg_log = '[the last %d iters], [loss_G %.5f], [D_A_losses %.5f], [D_B_losses %.5f],' \
                              % (iter_loss, G_losses[G_losses.__len__() - 1], D_A_losses[D_A_losses.__len__() - 1], \
                                 D_B_losses[D_B_losses.__len__() - 1])
                    print(avg_log)
                    open(log_path, 'a').write(avg_log + '\n')

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()

            # Save models checkpoints
            torch.save(netG_A2B.state_dict(), 'model_l/netG_A2B.pth')
            torch.save(netG_B2A.state_dict(), 'model_l/netG_B2A.pth')
            torch.save(netD_A.state_dict(), 'model_l/netD_A.pth')
            torch.save(netD_B.state_dict(), 'model_l/netD_B.pth')

            if (epoch + 1) % snapshot_epochs == 0:
                torch.save(netG_A2B.state_dict(), ('model_l/netG_A2B_%d.pth' % (epoch + 1)))
                torch.save(netG_B2A.state_dict(), ('model_l/netG_B2A_%d.pth' % (epoch + 1)))
                torch.save(netD_A.state_dict(), ('model_l/netD_A_%d.pth' % (epoch + 1)))
                torch.save(netD_B.state_dict(), ('model_l/netD_B_%d.pth' % (epoch + 1)))

            print('Epoch:{}'.format(epoch))


if __name__ == '__main__':
    sys.exit(main())
