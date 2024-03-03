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
from transformers import CLIPVisionModel, AutoProcessor
def torch_to_pil(torch_img):
    # torch_img = [3, H, W], from -1 to 1
    torch_img = torch_img.detach().cpu()
    np_img = np.array(((torch_img + 1) / 2) * 255).astype(np.uint8).transpose(1, 2, 0)
    pil = Image.fromarray(np_img)
    return pil

clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

def main(args):

    print(f' step 0. setting and file')
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32)
    set_determinism(args.seed)
    inference_dir = os.path.join(args.output_dir, 'image_generation_clip_conditioned')
    os.makedirs(inference_dir, exist_ok = True)

    print(f' step 0. call scratch model')
    device = torch.device("cuda")
    model = DiffusionModelUNet(spatial_dims=2,  # 2D Convolution
                               in_channels=3,  # input  RGB image
                               out_channels=3,  # output RGB image
                               num_channels=(128, 256, 256),  # 512
                               attention_levels=(False, False, True),
                               num_res_blocks=2,
                               num_head_channels=64,  # what is num_head_channels?
                               with_conditioning=True,
                               cross_attention_dim=768, )
    scheduler = DDIMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)

    print(f' step 1. pretrained dir')
    model_file = os.path.join(args.output_dir,'model')
    files = os.listdir(model_file)
    for file in files :
        file_name, ext = os.path.splitext(file)
        epoch = file_name.split('_')[-1]
        saved_state_dict = torch.load(os.path.join(model_file, file))
        org_state_dict = model.state_dict()
        for k in org_state_dict :
            org_state_dict[k] = saved_state_dict[f'module.{k}']
        model.load_state_dict(org_state_dict)
        model.to(device)

        # [2] for generation

        test_folder = r'/home/dreamyou070/MyData/anomaly_detection/MVTec/transistor/test'
        defects = os.listdir(test_folder)
        for defect in defects:
            defect_folder = os.path.join(test_folder, defect)
            rgb_folder = os.path.join(defect_folder, 'rgb')
            images = os.listdir(rgb_folder)
            for img in images:
                img_dir = os.path.join(rgb_folder, img)
                pil_img = Image.open(img_dir).convert('RGB')
                np_img = np.array(pil_img)
                inputs = clip_processor(images=np_img, return_tensors="pt")
                condition = clip_model(**inputs).last_hidden_state  # 1, 50, 768

                noise = torch.randn((1, 3, 256, 256)).to(device)

                # [3] generation test
                image, intermediates = inferer.sample(input_noise=noise,
                                                      diffusion_model=model,
                                                      scheduler=scheduler,
                                                      save_intermediates=True,
                                                      intermediate_steps=100,
                                                      condition = condition)
                b = image.shape[0]
                #for i, im in enumerate(intermediates) :
                img = image[0].squeeze()
                pil = torch_to_pil(img)
                pil.save(os.path.join(inference_dir, f'inference_epoch_{epoch}_condition_{defect}_{img}'))


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