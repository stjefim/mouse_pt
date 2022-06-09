import torch
import torch.nn as nn
import torch.optim as optim
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
NGPU = 0
BATCH = 128
EPOCHS = 30
LR = 0.0002
BETA1 = 0.5


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
            nn.ConvTranspose2d(NOISE_DIM, NGF * 8, 6, 1, 0, bias=False),
            nn.BatchNorm2d(NGF * 8),
            nn.ReLU(True),
            # state size. (NGF*8) x 6 x 6
            nn.ConvTranspose2d(NGF * 8, NGF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 4),
            nn.ReLU(True),
            # state size. (NGF*4) x 12 x 12
            nn.ConvTranspose2d(NGF * 4, NGF * 2, 5, 2, 1, bias=False),
            nn.BatchNorm2d(NGF * 2),
            nn.ReLU(True),
            # state size. (NGF*2) x 25 x 25
            nn.ConvTranspose2d(NGF * 2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            # state size. (NGF) x 50 x 50
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 100 x 100
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (NC) x 100 x 100
            nn.Conv2d(NC, NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF) x 50 x 50
            nn.Conv2d(NDF, NDF * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*2) x 25 x 25
            nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*4) x 13 x 13
            nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NDF * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (NDF*8) x 6 x 6
            nn.Conv2d(NDF * 8, 1, 6, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def train():
    dataset = dset.ImageFolder(root=DATA,
                            transform=transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()
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

    # Training Loop
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    for epoch in range(EPOCHS):
        for i, data in enumerate(dataloader, 0):
            pass
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
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'g': generator.state_dict(), 'd': discriminator.state_dict(), 'og': optimizerG.state_dict(),
                        'od': optimizerD.state_dict()}, f'cp/cp_{epoch}')



def test(checkpoint):
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

    generator = Generator(NGPU).to(device)
    if (device.type == 'cuda') and (NGPU > 1):
        generator = nn.DataParallel(generator, list(range(NGPU)))

    discriminator = Discriminator(NGPU).to(device)
    if (device.type == 'cuda') and (NGPU > 1):
        discriminator = nn.DataParallel(discriminator, list(range(NGPU)))

    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=device)

    optimizerD = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, 0.999))

    cp = torch.load(checkpoint, map_location=device)

    generator.load_state_dict(cp['g'])
    discriminator.load_state_dict(cp['d'])
    generator.eval()
    discriminator.eval()
    optimizerG.load_state_dict(cp['og'])
    optimizerD.load_state_dict(cp['od'])

    for i in range(10):
        plt.imshow(np.transpose(generator(fixed_noise)[0].detach().cpu().numpy()))
        plt.waitforbuttonpress()
            

if __name__ == '__main__':
    train()