import argparse
import random
from monai.utils import set_determinism
from data.prepare_dataset import call_dataset
import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from data.mvtec import passing_mvtec_argument

def main(args):

    print(f' step 1. seed seed')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_determinism(args.seed)

    print(f' step 2. make dataset')
    train_dataloader = call_dataset(args)

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
        images = torch.randn(1,3,256,256).to(device)
        optimizer.zero_grad(set_to_none=True)
        timesteps = torch.randint(0, 1000, (len(images),)).to(device)  # pick a random time step t
        with autocast(enabled=True):
            noise = torch.randn_like(images).to(device)
            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            loss = F.mse_loss(noise_pred.float(), noise.float())
            print(f'loss = {loss}')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Anomal DDPM')
    # step 1. setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='output')
    # step 2. dataset
    parser.add_argument('--data_path', type=str, default=r'../../../MyData/anomaly_detection/MVTec3D-AD')
    parser.add_argument("--anomal_source_path", type=str)
    parser.add_argument('--trigger_word', type=str)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--latent_res", type=int, default=64)
    parser.add_argument("--anomal_only_on_object", action='store_true')
    parser.add_argument("--do_anomal_sample", action='store_true')
    parser.add_argument("--do_object_detection", action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument("--anomal_p", type=float, default=0.04)
    parser.add_argument('--obj_name', type=str, default='bottle')
    parser.add_argument("--reference_check", action='store_true')
    parser.add_argument("--use_small_anomal", action='store_true')
    parser.add_argument("--anomal_min_perlin_scale", type=int, default=0)
    parser.add_argument("--anomal_max_perlin_scale", type=int, default=3)
    parser.add_argument("--anomal_min_beta_scale", type=float, default=0.5)
    parser.add_argument("--anomal_max_beta_scale", type=float, default=0.8)
    parser.add_argument("--back_min_perlin_scale", type=int, default=0)
    parser.add_argument("--back_max_perlin_scale", type=int, default=3)
    parser.add_argument("--back_min_beta_scale", type=float, default=0.6)
    parser.add_argument("--back_max_beta_scale", type=float, default=0.9)
    parser.add_argument("--do_rot_augment", action='store_true')
    parser.add_argument("--anomal_trg_beta", type=float)
    parser.add_argument("--back_trg_beta", type=float)

    args = parser.parse_args()
    passing_mvtec_argument(args)
    main(args)