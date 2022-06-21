import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import random


torch.manual_seed(999)
random.seed(999)

DATA = 'data/'
NOISE_DIM = 100
NGF = 64
NDF = 64
NC = 1
NGPU = 1
BATCH = 128
EPOCHS = 100
LR = 0.0002
BETA1 = 0.5


def log(msg, file):
    print(msg)
    print(msg, file=file)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NOISE_DIM, NGF * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # state size. (NGF*8) x 4 x 4
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # state size. (NGF*4) x 8 x 8
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # state size. (NGF*2) x 16 x 16
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # state size. (NGF) x 32 x 32
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF) x 32 x 32
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*2) x 16 x 16
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*4) x 8 x 8
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 4 x 4
            nn.Conv2d(NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



dataset = dset.ImageFolder(root=DATA,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Grayscale()
                            ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH,
                                        shuffle=True, num_workers=2)
device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

generator = Generator(NGPU).to(device)
if (device.type == 'cuda') and (NGPU > 1):
    generator = nn.DataParllel(generator, list(range(NGPU)))
generator.apply(weights_init)

discriminator = Discriminator(NGPU).to(device)
if (device.type == 'cuda') and (NGPU > 1):
    discriminator = nn.DataParallel(discriminator, list(range(NGPU)))
discriminator.apply(weights_init)

criterion = nn.BCELoss()
fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=device)

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, 0.999))
cur_epoch = 0




def train():
    f = open('log.txt', 'w')
    generator.train()
    discriminator.train()
    print("Starting Training Loop...")
    for epoch in range(cur_epoch + 1, EPOCHS):
        for i, data in enumerate(dataloader, 0):
            ###########################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ##########################

            # Train with all-real batch
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, NOISE_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            label.fill_(real_label)
            output = discriminator(real_cpu).view(-1)
            D_x_2 = output.mean().item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                log('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x):  %.4f / %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_x_2, D_G_z1, D_G_z2), f)


            if epoch % 5 == 0: 
                torch.save({'epoch': epoch, 'g': generator.state_dict(), 'd': discriminator.state_dict(), 'og': optimizerG.state_dict(),
                         'od': optimizerD.state_dict()}, f'cp/cp_{epoch}')
    f.close()

def load(checkpoint):
    cp = torch.load(checkpoint, map_location=device)
    global cur_epoch
    cur_epoch = cp['epoch']
    generator.load_state_dict(cp['g'])
    discriminator.load_state_dict(cp['d'])
    optimizerG.load_state_dict(cp['og'])
    optimizerD.load_state_dict(cp['od'])


def test():
    generator.eval()
    discriminator.eval()
    for i in range(10):
        imgs = generator(fixed_noise)
        plt.imshow(np.transpose(imgs[i].detach().cpu().numpy()), cmap='Greys_r')
        plt.title(str(discriminator(imgs)[i]))
        plt.waitforbuttonpress()
            

if __name__ == '__main__':
    load('cp/cp_75')
    test()
