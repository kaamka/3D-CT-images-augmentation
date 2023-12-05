# Imports

import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from monai.networks.layers import Reshape

from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Compose,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ScaleIntensityRange,
    EnsureType,
    Transform,
    Resize,
    CenterSpatialCrop,
    AdjustContrast
)

from monai.data import CacheDataset, DataLoader, MetaTensor
from monai.visualize import matshow3d, img2tensorboard
from monai.utils import first, set_determinism
from monai.apps import get_logger

# Define run name and paths

RESUME_TRAINING = True # if set to TRUE provide run_name to continue
run_name = '04-12-2023_16:48'

# Due to issues with running training lately, when continuing training save logs and models in temp directory and if 
# training finished correctly, manually copy them (or automatically at the end of script)
save_path = '/data2/etude/micorl/WGAN'

if not RESUME_TRAINING:
    run_name = datetime.now().strftime("%d-%m-%Y_%H:%M")


logs_path = os.path.join(save_path, 'logs/', run_name)

checkpoint_name = f'checkpoint_{run_name}.pt'
checkpoint_path = os.path.join(save_path, 'models/', checkpoint_name)

print("logs_path: " + logs_path)
print("checkpoint_path: " + checkpoint_path)

# Determinism, device and logger

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
get_logger('train_log')
set_determinism(0)
device = torch.device('cuda:0')

torch.cuda.empty_cache()
torch.cuda.memory_stats()
# print(torch.cuda.memory_summary())

# Load and prepare dataset

image_size = 256
num_slices = 32
contrast_gamma = 1.5
every_n_slice = 1
batch_size = 4

learning_rate = 1e-5
num_epochs_list = [3000] # 250, 250, 1500, 1000, 1000
latent_size = 100
critic_iterations_list = [1] # 5, 3, 1, 1
generator_iterations_list = [3] # 1, 1, 1, 3
lambda_gp = 10 # controls how much of gradient penalty will be added to critic loss

assert len(num_epochs_list) == len(critic_iterations_list)
assert len(num_epochs_list) == len(generator_iterations_list)

win_wid = 400
win_lev = 60

class ReduceDepth(Transform):
    def __init__(self, start_slice, end_slice):
        self.start_slice = start_slice
        self.end_slice = end_slice

    def __call__(self, data):
        t = data.get_array()
        t = t[:, :, :, self.start_slice:self.end_slice+1]
        data.set_array(t)
        return data

data_dir = '/data1/dose-3d-generative/data_med/PREPARED/FOR_SEG'
directory = os.path.join(data_dir, 'ct_images_prostate_32fixed')
images_pattern = os.path.join(directory, '*.nii.gz')
images = sorted(glob.glob(images_pattern))

train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        CenterSpatialCrop((380, 380, 0)),
        Resize((image_size, image_size, num_slices)),
        ScaleIntensityRange(a_min=win_lev-(win_wid/2), a_max=win_lev+(win_wid/2), b_min=0.0, b_max=1.0, clip=True),
        RandRotate(range_x=np.pi/12, prob=0.5, keep_size=True),
        # RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType()
    ]
)

dataset = CacheDataset(images, train_transforms)
loader = torch.utils.data.DataLoader(dataset, num_workers=10, shuffle=True, pin_memory=torch.cuda.is_available(), batch_size=batch_size)

image_sample = first(loader)
print(image_sample.shape)

# with torch.no_grad():
#     fig = plt.figure(figsize=(15,15))
#     matshow3d(volume=image_sample[0, 0, :, :, :],
#             fig=fig,
#             title="Loaded image",
#             every_n=every_n_slice,
#             frame_dim=-1,
#             cmap="gray")
#     fig.savefig('test.png')

# Define model architecture

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.net = nn.Sequential(
            self._block(1, 8, 4, 2, 1),
            self._block(8, 32, 4, 2, 1),
            self._block(32, 64, 4, 2, 1),
            self._block(64, 128, 4, 2, 1),        
            nn.Conv3d(128, 1, kernel_size=3, stride=(2,2,1), padding=1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear = nn.Linear(100, 256*8*8*4)
        self.reshape = Reshape(256, 8, 8, 4)

        self.net = nn.Sequential(
            self._block(256, 128, 4, 2, 1),
            self._block(128, 64, 4, 2, 1),
            self._block(64, 32, 4, 2, 1),
            self._block(32, 16, (4,4,3), (2,2,1), 1),
            self._block(16, 1, (4,4,3), (2,2,1), 1),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.reshape(x)
        x = self.net(x)
        return x

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.BatchNorm3d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic, real, fake, device='cpu'):
    batch_size, c, h, w, d = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1, 1)).repeat(1, c, h, w, d).to(device)

    # calculate interpolated/mean images using random vector alpha and 1-alpha
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    #print('1 ', mixed_scores.shape)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # reshape gradient to vector, calculate norm
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    
    # calculate gradient penalty
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty

def test():
    N, in_channels, H, W, D = 1, 1, image_size, image_size, 32
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W, D))
    disc = Critic()
    assert disc(x).shape == (N, 1, 8, 8, 2), "Discriminator test failed"
    gen = Generator()
    z = torch.randn(1, noise_dim)
    assert gen(z).shape == (N, in_channels, H, W, D), "Generator test failed"
    print("Success, tests passed!")

test()

# Initialize and train model

# create and initialize networks
critic = Critic().to(device)
initialize_weights(critic)

generator = Generator().to(device)
initialize_weights(generator)

# initialize optimizers
opt_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

# initialize schedulers
# TODO use stepLR instead
# scheduler_critic = optim.lr_scheduler.LinearLR(opt_critic, start_factor=1.0, end_factor=0.05, total_iters=35000)
# scheduler_generator = optim.lr_scheduler.LinearLR(opt_generator, start_factor=1.0, end_factor=0.05, total_iters=35000)

# tensorboard writers
writer_real = SummaryWriter(logs_path + '/real')
writer_fake = SummaryWriter(logs_path + '/fake')
writer_real_gif = SummaryWriter(logs_path + '/real_gif')
writer_fake_gif = SummaryWriter(logs_path + '/fake_gif')
writer_loss_step = SummaryWriter(logs_path + '/loss_step')
writer_loss_epoch = SummaryWriter(logs_path + '/loss_epoch')

def save_model(epoch, step):
    state = {
        'epoch': epoch + 1,
        'step': step,
        'generator': generator.state_dict(),
        'critic': critic.state_dict(),
        'opt_generator': opt_generator.state_dict(),
        'opt_critic': opt_critic.state_dict(),
    }

    torch.save(state, checkpoint_path)

# load state if resuming training
if RESUME_TRAINING:
    state = torch.load(checkpoint_path)
    critic.load_state_dict(state['critic'])
    generator.load_state_dict(state['generator'])
    opt_critic.load_state_dict(state['opt_critic'])
    opt_generator.load_state_dict(state['opt_generator'])
    step = state['step']
    start_epoch = state['epoch']
    end_epoch = start_epoch + num_epochs_list[0]
    all_epochs = start_epoch + sum(num_epochs_list)
else:
    step = 0
    start_epoch = 0
    all_epochs = sum(num_epochs_list)

critic.train()
generator.train()

for num_epochs, critic_iterations, generator_iterations in zip(num_epochs_list, critic_iterations_list, generator_iterations_list):
    end_epoch = start_epoch + num_epochs
    print(f'num_epochs = {num_epochs}, critic_iterations = {critic_iterations}, generator_iterations = {generator_iterations}, start_epoch = {start_epoch}, end_epoch = {end_epoch}')
    for epoch in range(start_epoch, end_epoch):
        curr_epoch_loss_gen_sum = 0
        curr_epoch_loss_crit_sum = 0
        for batch_idx, real in enumerate(loader):
            real = real.to(device)
            cur_batch_size = real.shape[0]

            # Train Critic: max E[critic(real)] - E[critic(fake)]
            # equivalent to minimizing the negative of that
            for _ in range(critic_iterations):
                noise = torch.randn(cur_batch_size, latent_size).to(device)
                fake = generator(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, device=device)
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                )
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()


            for _ in range(generator_iterations):
            # # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
                generator_fake = critic(fake).reshape(-1)
                loss_generator = -torch.mean(generator_fake)
                generator.zero_grad()
                loss_generator.backward()
                opt_generator.step()
                # scheduler_critic.step()
                # scheduler_generator.step()
                writer_loss_step.add_scalar("Step loss Generator", loss_generator, global_step=step)
                writer_loss_step.add_scalar("Step loss Crit", loss_critic, global_step=step)
                curr_epoch_loss_gen_sum += loss_generator.item()
                curr_epoch_loss_crit_sum += loss_critic.item()
                noise = torch.randn(cur_batch_size, latent_size).to(device)
                fake = generator(noise)

            # Print losses occasionally and print to tensorboard
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{all_epochs}] Batch {batch_idx}/{len(loader)} \
                    Loss D: {loss_critic:.4f}, loss G: {loss_generator:.4f}"
                )

                with torch.no_grad():
                    fake = generator(noise)
                    fig_real = plt.figure(figsize=(15,15))
                    matshow3d(volume=real, fig=fig_real, title="real", every_n=2, frame_dim=-1, cmap="gray")
                    fig_fake = plt.figure(figsize=(15,15))
                    matshow3d(volume=fake, fig=fig_fake, title="fake", every_n=2, frame_dim=-1, cmap="gray")
                    img2tensorboard.plot_2d_or_3d_image(data=real, step=step, writer=writer_real_gif, frame_dim=-1)
                    img2tensorboard.plot_2d_or_3d_image(data=fake, step=step, writer=writer_fake_gif, frame_dim=-1)
                    
                    writer_real.add_figure("Real", fig_real, global_step=step)
                    writer_fake.add_figure("Fake", fig_fake, global_step=step)

            step += 1

        if epoch % 10 == 0:
            save_model(epoch, step)

        loss_gen_epoch = curr_epoch_loss_gen_sum / len(loader)
        loss_crit_epoch = curr_epoch_loss_crit_sum / len(loader)
        print(f"Generator mean loss in epoch:{loss_gen_epoch}")
        writer_loss_epoch.add_scalar("Epoch loss Generator", loss_gen_epoch, global_step=epoch)
        writer_loss_epoch.add_scalar("Epoch loss Crit", loss_crit_epoch, global_step=epoch)
    
    start_epoch = end_epoch

noise = torch.randn(1, latent_size).to(device)
with torch.no_grad():
    fake = generator(noise)
    fig = plt.figure(figsize=(15,15))
    matshow3d(volume=fake,
            fig=fig,
            title="Generated image",
            every_n=every_n_slice,
            frame_dim=-1,
            cmap="gray")
    fig.savefig('test.png')

save_model(epoch, step)
