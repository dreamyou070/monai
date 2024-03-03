import argparse
import random
from monai.utils import set_determinism
from data.prepare_dataset import call_dataset
import os
import torch
import json
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from data.mvtec import passing_mvtec_argument
from tqdm import tqdm

def main(args):

    print(f' step 0. base path')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    record_save_dir = os.path.join(output_dir, 'record')
    os.makedirs(record_save_dir, exist_ok=True)
    with open(os.path.join(record_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    model_base_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_base_dir, exist_ok = True)

    print(f' step 1. seed seed')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_determinism(args.seed)

    print(f' step 2. make dataset')
    train_dataloader = call_dataset(args, is_valid = False)

    print(f' step 3. model and scheduler')
    device = torch.device("cuda")
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    model = DiffusionModelUNet(spatial_dims=2,  # 2D Convolution
                               in_channels=3,   # input  RGB image
                               out_channels=3,  # output RGB image
                               num_channels=(320, 640, 1280), # 512
                               attention_levels=(False, False, True),
                               num_res_blocks=2,
                               num_head_channels=64, # what is num_head_channels?
                               with_conditioning=True,
                               cross_attention_dim = 768,)
    model.to(device)
    inferer = DiffusionInferer(scheduler)

    print(f' step 4. optimizer')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    print(f' step 5. Training')
    # [0] progress bar
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, desc="steps")
    global_step = 0
    scaler = GradScaler()
    loss_dict = {}
    for epoch in range(args.start_epoch, args.max_train_epochs + args.start_epoch):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            # [1] call image
            image = batch['image'].to(device)
            condition = torch.randn(1,50,768).to(device).to(image.dtype)  # why don't use generated image
            timesteps = torch.randint(0, 1000, (len(image),)).to(device)  # pick a random time step t
            with autocast(enabled=True):
                noise = torch.randn_like(image).to(device)
                # Get model prediction
                noise_pred = inferer(inputs=image, diffusion_model=model, noise=noise, timesteps=timesteps,
                                     condition = condition)


                loss = F.mse_loss(noise_pred.float(), noise.float())
                loss_dict['loss'] = loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            # [3] progress bar
            progress_bar.update(1)
            global_step += 1
            progress_bar.set_postfix(**loss_dict)
        # [2] save model per epoch
        torch.save(model.state_dict(), os.path.join(model_base_dir, f'model_{epoch+1}.pth'))


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
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--on_desktop", action='store_true')
    args = parser.parse_args()
    passing_mvtec_argument(args)
    main(args)