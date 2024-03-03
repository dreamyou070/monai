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
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet, DiffusionModelEncoder
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

    print(f' step 4. classifier')
    classifier = DiffusionModelEncoder(spatial_dims=2,
                                       in_channels=3,
                                       out_channels=2,                                  # normal and anomal classify
                                       num_channels=(32, 64, 64),
                                       attention_levels=(False, True, True),
                                       num_res_blocks=(1, 1, 1),
                                       num_head_channels=64,
                                       with_conditioning=False,)
    classifier.to(device)

    """

    print(f' step 4. optimizer')
    optimizer_cls = torch.optim.Adam(params=classifier.parameters(), lr=2.5e-5)

    print(f' step 5. classifier training') # as if local model training
    # [0] progress bar
    max_train_steps = len(train_dataloader) * args.max_train_epochs
    progress_bar = tqdm(range(max_train_steps), smoothing=0, desc="steps")
    global_step = 0
    scaler = GradScaler()
    loss_dict = {}
    for epoch in range(args.start_epoch, args.max_train_epochs + args.start_epoch):
        classifier.train()
        epoch_loss = 0
        for step, data in enumerate(train_dataloader):
            images = data["image"].to(device)
            classes = data["slice_label"].to(device)
            # classes[classes==2]=0
            optimizer_cls.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (len(images),)).to(device)

            with autocast(enabled=False):
                noise = torch.randn_like(images).to(device)
                noisy_img = scheduler.add_noise(images, noise, timesteps)  # add t steps of noise to the input image
                # classifier classify image with timestep
                pred = classifier(noisy_img, timesteps)
                # cross entropy loss
                loss = F.cross_entropy(pred, classes.long())
                loss.backward()
                optimizer_cls.step()
            epoch_loss += loss.item()
    """
    """
    print(f' step 6. Noising Input Image')  # as if local model training
    L = 200
    current_img = torch.randn(1,3,64,64)
    scheduler.set_timesteps(num_inference_steps=1000)
    progress_bar = tqdm(range(L))  # go back and forth L timesteps
    for t in progress_bar:
        with autocast(enabled=False):
            with torch.no_grad():
                model_output = model(current_img, timesteps=torch.Tensor((t,)).to(current_img.device))
        current_img, _ = scheduler.reversed_step(model_output, t, current_img)
    """


    print(f' Step 7. Reconstruct using Classifier Gradient')
    y = torch.tensor(0)  # define the desired class label (normal)
    scale = 6  # define the desired gradient scale s
    L = 200
    progress_bar = tqdm(range(L))  # go back and forth L timesteps
    current_img = torch.randn(1,3,64,64).to(device)
    for i in progress_bar:  # go through the denoising process
        t = L - i
        with autocast(enabled=True):
            with torch.no_grad():
                # (1,3,64,64)
                model_output = model(current_img, timesteps=torch.Tensor((t,)).to(current_img.device)).detach()  # this is supposed to be epsilon
            with torch.enable_grad():
                x_in = current_img.detach().requires_grad_(True)
                # generate probability (batch, 2) -> what is 2 means ?
                logits = classifier(x_in, timesteps=torch.Tensor((t,)).to(current_img.device))
                log_probs = F.log_softmax(logits, dim=-1)
                print(f'log_probs = {log_probs}')
                selected = log_probs[range(len(logits)), y.view(-1)] # select the first one
                print(f'selected = {selected}')
                a = torch.autograd.grad(selected.sum(), x_in)[0] # scaling following score
                alpha_prod_t = scheduler.alphas_cumprod[t]
                updated_noise = (model_output - (1 - alpha_prod_t).sqrt() * scale * a)  # update the predicted noise epsilon with the gradient of the classifier
        # [4] next step predicting
        current_img, _ = scheduler.step(updated_noise,
                                        t,
                                        current_img)
        torch.cuda.empty_cache()




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