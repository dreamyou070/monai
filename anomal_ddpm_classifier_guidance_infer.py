import argparse
import random
from monai.utils import set_determinism
import os
import torch
from generative.inferers import DiffusionInferer
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.schedulers.ddim import DDIMScheduler
from data.mvtec import passing_mvtec_argument
import numpy as np
from PIL import Image

def torch_to_pil(torch_img):
    # torch_img = [3, H, W], from -1 to 1
    np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
    pil = Image.fromarray(np_img)
    return pil

def main(args):

    print(f' step 0. setting and file')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_determinism(args.seed)
    inference_dir = os.path.join(args.output_dir, 'image_generation')
    os.makedirs(inference_dir, exist_ok = True)

    print(f' step 0. call scratch model')
    device = torch.device("cuda")
    model = DiffusionModelUNet(spatial_dims=2,  # 2D Convolution
                               in_channels=3,  # input  RGB image
                               out_channels=3,  # output RGB image
                               num_channels=(64, 64, 64),
                               attention_levels=(False, False, True),
                               num_res_blocks=1,
                               num_head_channels=64,  # what is num_head_channels?
                               with_conditioning=False, )
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)

    print(f' step 1. pretrained dir')
    model_file = os.path.join(args.output_dir,'model')
    files = os.listdir(model_file)
    for file in files :
        file_name, ext = os.path.splitext(file)
        epoch = file_name.split('_')[-1]
        model.load_state_dict(torch.load(os.path.join(model_file, file)))
        model.to(device)

        # [2] for generation
        noise = torch.randn((1, 3, 64, 64)).to(device)

        # [3] generation test
        image, intermediates = inferer.sample(input_noise=noise,
                                              diffusion_model=model,
                                              scheduler=scheduler,
                                              save_intermediates=True,
                                              intermediate_steps=100)
        b = image.shape[0]
        for b_idx in range(b) :
            img = image[b_idx].squeeze()
            pil = torch_to_pil(img)
            pil.save(os.path.join(inference_dir, f'inference_epoch_{epoch}.png'))



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Anomal DDPM Inference')
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