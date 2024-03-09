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
from data.dataset import passing_mvtec_argument
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
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    model = DiffusionModelUNet(spatial_dims=2,  # 2D Convolution
                               in_channels=3,   # input  RGB image
                               out_channels=3,  # output RGB image
                               num_channels=(128, 256, 256), # 512
                               attention_levels=(True, True, True),
                               num_res_blocks=2,
                               num_head_channels=64, # what is num_head_channels?
                               with_conditioning=True, # cross attention
                               cross_attention_dim = 768,)
    #inferer = DiffusionInferer(scheduler)

    print(f' step 4. optimizer')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    print(f' step 5. detection')
    anomal_detection = torch.nn.parameter.Parameter(data = torch.zeros(1,768),
                                                    requires_grad=True)

    print(f'\n step 9. registering saving tensor')
    from attention_store import AttentionStore
    from attention_store.attention_control import register_attention_control
    controller = AttentionStore()
    register_attention_control(model, controller)

    print(f' step 5. prepare accelerator')
    from utils.accelerator_utils import prepare_accelerator
    args.logging_dir = os.path.join(output_dir, 'logging')
    accelerator = prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    print(f' step 6. model to accelerator')
    train_dataloader, model, optimizer = accelerator.prepare(train_dataloader, model, optimizer)

    print(f' step 7. Training')
    # [0] progress bar
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, desc="steps")
    global_step = 0
    loss_dict = {}
    for epoch in range(args.start_epoch, args.max_train_epochs + args.start_epoch):
        model.train()
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            # [1] call image
            image = batch['image']
            b_size = image.shape[0]
            # [2] condition
            anomal_detection = anomal_detection.unsqueeze(0).repeat(b_size, 1, 1)
            with autocast(enabled=True):
                model(x= image,
                      timestep=0,
                      context = anomal_detection,
                      down_block_additional_residuals = None,
                      mid_block_additional_residual  = None)



            #accelerator.backward(loss)
            #optimizer.step()
            #epoch_loss += loss.item()
            # [3] progress bar
            if is_main_process :
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(**loss_dict)
        # [2] save model per epoch

        if is_main_process :
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
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], )
    parser.add_argument("--save_precision", type=str, default=None, choices=[None, "float", "fp16", "bf16"], )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--log_with", type=str, default=None, choices=["tensorboard", "wandb", "all"], )
    parser.add_argument("--log_prefix", type=str, default=None)
    parser.add_argument("--lowram", action="store_true", )
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
    parser.add_argument("--clip_test", action='store_true')
    args = parser.parse_args()
    passing_mvtec_argument(args)
    main(args)