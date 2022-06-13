import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as dsets


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,))])
# MNIST dataset
mnist = datasets.MNIST(root='./data/',
                       train=True,
                       transform=transform,
                       download=True)
# Data loader
# data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=100, shuffle=True)

# todo: this fails with:  Dataset not found or corrupted
train_dataset = dsets.CelebA(root='C:\Studies\DeepLearning\ex3_data',
                             split='train',
                             transform=transform,
                             download=False)  # make sure you set it to False

data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=100, shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(784, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())

# Generator
G = nn.Sequential(
    nn.Linear(64, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 784),
    nn.Tanh())

if torch.cuda.is_available():
    D.cuda()
    G.cuda()

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

for epoch in range(200):
    accuracy_real = 0
    accuracy_fake = 0

    for i, (images, _) in enumerate(data_loader):
        # Build mini-batch dataset
        batch_size = images.size(0)
        images = to_cuda(images.view(batch_size, -1))

        # Create the labels which are later used as input for the BCE loss
        real_labels = to_cuda(torch.ones(batch_size))
        fake_labels = to_cuda(torch.zeros(batch_size))

        # ============= Train the discriminator =============#
        # Compute BCE_Loss using real images where BCE_Loss(x, y):
        #         - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        D.train()
        G.train(False)  # <-> G.eval()

        outputs = D(images)  # Real images
        d_loss_real = criterion(outputs.squeeze(1), real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = to_cuda(torch.randn(batch_size, 64))
        fake_images = G(z)  # Generate fake images
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs.squeeze(1), fake_labels)
        fake_score = outputs

        # Backprop + Optimize
        d_loss = d_loss_real + d_loss_fake
        D.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # =============== Train the generator ===============#
        # Compute loss with fake images
        D.train(False)
        G.train()
        z = to_cuda(torch.randn(batch_size, 64))
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        g_loss = criterion(outputs.squeeze(1), real_labels)

        # Backprop + Optimize
        D.zero_grad()
        G.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 300 == 0:
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                  'g_loss: %.4f, D(x): %.2f, D(G(z)): %.2f'
                  % (epoch, 200, i + 1, 600, d_loss.data, g_loss.data,
                     real_score.data.mean(), fake_score.data.mean()))

    # Save real images
    if (epoch + 1) == 1:
        images = images.view(images.size(0), 1, 28, 28)
        save_image(denorm(images.data), './data/real_images.png')

    plt.imshow(denorm(fake_images.data[0]).view(28, 28).cpu().numpy(), cmap='gray')
    plt.show()

    # Save sampled images
    fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
    save_image(denorm(fake_images.data), './data/fake_images-%d.png' % (epoch + 1))
