import time
import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
import argparse


def main(args):

    print(f' step 1. seed seed')
    set_determinism(args.seed)

    channel = 0  # 0 = Flair
    assert channel in [0, 1, 2, 3], "Choose a valid channel"

    print(f' step 3. model and scheduler')
    device = torch.device("cuda")
    model = DiffusionModelUNet(spatial_dims=2,  # 2D Convolution
                               in_channels=3,   # input  RGB image
                               out_channels=3,  # output RGB image
                               num_channels=(64, 64, 64),
                               attention_levels=(False, False, True),
                               num_res_blocks=1,
                               num_head_channels=64, # what is num_head_channels?
                               with_conditioning=False,)
    model.to(device)
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)

    print(f' step 4. optimizer')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    print(f' step 5. Training')
    n_epochs = 2000
    val_interval = 20
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        images = torch.randn(1,3,256,256)
        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t
        with autocast(enabled=True):
            noise = torch.randn_like(images).to(device)
            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Anomal DDPM')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    main(args)