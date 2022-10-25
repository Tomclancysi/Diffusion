import torch
from torch.utils.data import DataLoader
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, Unet, Trainer, Dataset, DataLoader


# dataset = Dataset(folder="/workspace/ln/PyTorch-VAE/Data/aqi", image_size=64)
# loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '/workspace/ln/PyTorch-VAE/Data/aqi',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

trainer.train()
