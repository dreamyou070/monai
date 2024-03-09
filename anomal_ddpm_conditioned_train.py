import argparse
import random
from monai.utils import set_determinism
from data.prepare_dataset import call_dataset
import os
import torch
import json
from torch import nn
from attention_store.normal_activator import NormalActivator
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
    train_dataloader = call_dataset(args)

    print(f' step 3. model and scheduler')
    #scheduler = DDIMScheduler(num_train_timesteps=1000)
    model = DiffusionModelUNet(spatial_dims=2,  # 2D Convolution
                               in_channels=3,   # input  RGB image
                               out_channels=3,  # output RGB image
                               num_channels=(64, 64, 128), # 512
                               attention_levels=(True, True, True),
                               num_res_blocks=2,
                               num_head_channels=64, # what is num_head_channels?
                               with_conditioning=True, # cross attention
                               cross_attention_dim = 768,)
    #inferer = DiffusionInferer(scheduler)


    print(f' step 5. detection')
    anomal_detection = torch.nn.parameter.Parameter(data = torch.zeros(1,768),
                                                    requires_grad=True)
    trainable_params = []
    trainable_params.append({"params": model.parameters(), "lr": args.learning_rate})
    trainable_params.append({"params": anomal_detection, "lr": args.learning_rate})

    print(f' step 4. optimizer')
    optimizer = torch.optim.Adam(trainable_params)

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
    train_dataloader, model, anomal_detection, optimizer = accelerator.prepare(train_dataloader, model,
                                                                               anomal_detection,
                                                                               optimizer)

    print(f'\n step 7. loss function')
    loss_focal = None
    loss_l2 = torch.nn.modules.loss.MSELoss(reduction='none')
    normal_activator = NormalActivator(loss_focal, loss_l2)

    print(f' step 7. Training')
    def resize_query_features(query):

        head_num, pix_num, dim = query.shape
        res = int(pix_num ** 0.5)  # 8
        query_map = query.view(head_num, res, res, dim).permute(0, 3, 1, 2).contiguous()  # 1, channel, res, res
        resized_query_map = nn.functional.interpolate(query_map, size=(64, 64), mode='bilinear')  # 1, channel, 64,  64
        resized_query = resized_query_map.permute(0, 2, 3, 1).contiguous().squeeze()  # head, 64, 64, channel
        resized_query = resized_query.view(head_num, 64 * 64, dim)  # 8, 64*64, dim
        return resized_query

    # [0] progress bar
    args.max_train_steps = len(train_dataloader) * args.max_train_epochs
    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, desc="steps")
    global_step = 0
    loss_dict = {}
    loss_list = []
    weight_dtype = torch.float32
    for epoch in range(args.start_epoch, args.max_train_epochs + args.start_epoch):

        model.train()
        epoch_loss_total = 0
        device = accelerator.device
        loss = torch.tensor(0.0, dtype=weight_dtype, device=accelerator.device)
        loss_dict = {}

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            # [1] call image
            image = batch['image']
            gt = batch['gt']
            anomal_position_vector = gt.squeeze().flatten()
            b_size = image.shape[0]
            # [2] condition
            anomal_detection = anomal_detection.unsqueeze(0).repeat(b_size, 1, 1)
            with autocast(enabled=True):
                model(x= image,
                      timesteps=torch.Tensor([0]),
                      context = anomal_detection,
                      down_block_additional_residuals = None,
                      mid_block_additional_residual  = None)
            query_dict, key_dict, attn_dict = controller.query_dict, controller.key_dict, controller.attn_dict
            controller.reset()
            attn_list, origin_query_list, query_list, key_list = [], [], [], []
            for layer in args.trg_layer_list:
                query = query_dict[layer][0].squeeze()  # head, pix_num, dim
                origin_query_list.append(query)  # head, pix_num, dim
                query_list.append(resize_query_features(query))  # head, pix_num, dim
                key_list.append(key_dict[layer][0])  # head, pix_num, dim
                # attn_list.append(attn_dict[layer][0])
            # [1] local
            local_query = torch.cat(query_list, dim=-1)  # head, pix_num, long_dim
            local_key = torch.cat(key_list, dim=-1).squeeze()  # head, 77, long_dim
            # learnable weight ?
            # local_query = [8, 64*64, 280] = [64*64, 2240]
            attention_scores = torch.baddbmm(
                torch.empty(local_query.shape[0], local_query.shape[1], local_key.shape[1], dtype=query.dtype,
                            device=query.device), local_query, local_key.transpose(-1, -2), beta=0, )
            local_attn = attention_scores.softmax(dim=-1)[:, :, :2]
            if args.normal_activating_test:
                normal_activator.collect_attention_scores(local_attn, anomal_position_vector,  # anomal position
                                                          1 - anomal_position_vector, False)
            else:
                normal_activator.collect_attention_scores(local_attn, anomal_position_vector,  # anomal position
                                                          1 - anomal_position_vector, True)
            normal_activator.collect_anomal_map_loss(local_attn, anomal_position_vector, )

            # [5] backprop
            if args.do_attn_loss:
                normal_cls_loss, normal_trigger_loss, anomal_cls_loss, anomal_trigger_loss = normal_activator.generate_attention_loss()
                if type(anomal_cls_loss) == float:
                    attn_loss = args.normal_weight * normal_trigger_loss.mean()
                else:
                    attn_loss = args.normal_weight * normal_cls_loss.mean() + args.anomal_weight * anomal_cls_loss.mean()
                if args.do_cls_train:
                    if type(anomal_trigger_loss) == float:
                        attn_loss = args.normal_weight * normal_cls_loss.mean()
                    else:
                        attn_loss += args.normal_weight * normal_cls_loss.mean() + args.anomal_weight * anomal_cls_loss.mean()
                loss += attn_loss
                loss_dict['attn_loss'] = attn_loss.item()

            if args.do_map_loss:
                map_loss = normal_activator.generate_anomal_map_loss()
                loss += map_loss
                loss_dict['map_loss'] = map_loss.item()

            if args.test_noise_predicting_task_loss:
                noise_pred_loss = normal_activator.generate_noise_prediction_loss()
                loss += noise_pred_loss
                loss_dict['noise_pred_loss'] = noise_pred_loss.item()


            loss = loss.to(weight_dtype)
            current_loss = loss.detach().item()
            if epoch == args.start_epoch:
                loss_list.append(current_loss)
            else:
                epoch_loss_total -= loss_list[step]
                loss_list[step] = current_loss
            epoch_loss_total += current_loss
            avr_loss = epoch_loss_total / len(loss_list)
            loss_dict['avr_loss'] = avr_loss
            loss_dict['sample'] = batch['is_ok']  # if 1 = normal sample, if 0 = anormal sample
            accelerator.backward(loss)
            optimizer.step()
            #lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            if is_main_process:
                progress_bar.set_postfix(**loss_dict)
            normal_activator.reset()
            if global_step >= args.max_train_steps:
                break
            # ----------------------------------------------------------------------------------------------------------- #
            # [6] epoch final

        accelerator.wait_for_everyone()
        """
        #if is_main_process:
            #ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            #save_model(args, ckpt_name, accelerator.unwrap_model(network), save_dtype)
            
            if position_embedder is not None:
                position_embedder_base_save_dir = os.path.join(args.output_dir, 'position_embedder')
                os.makedirs(position_embedder_base_save_dir, exist_ok=True)
                p_save_dir = os.path.join(position_embedder_base_save_dir,
                                          f'position_embedder_{epoch + 1}.safetensors')
                pe_model_save(accelerator.unwrap_model(position_embedder), save_dtype, p_save_dir)
        """

    #accelerator.end_training()

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
    parser.add_argument("--min_perlin_scale", type=int, default=0)
    parser.add_argument("--max_perlin_scale", type=int, default=3)
    parser.add_argument("--min_beta_scale", type=float, default=0.5)
    parser.add_argument("--max_beta_scale", type=float, default=0.8)
    parser.add_argument("--do_rot_augment", action='store_true')
    parser.add_argument("--trg_beta", type=float)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--max_train_epochs", type=int, default=None, )
    parser.add_argument("--on_desktop", action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--rgb_train', action='store_true')
    parser.add_argument('--do_self_aug', action='store_true')
    args = parser.parse_args()
    passing_mvtec_argument(args)
    main(args)