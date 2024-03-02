# !/bin/bash
# scratch vae / trained vae
# matching losses : only normal

port_number=50005
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="from_trained_pe_local_global_from_pretrained_vae_global_matching"

anomal_source_path="../../../MyData/anomal_source"
network_weights="../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/local_all_crossattn_pe/models/epoch-000009.safetensors" \

python --main_process_port $port_number ../anomal_ddpm_classifier_guidance.py
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --train_unet --train_text_encoder --start_epoch 0 --max_train_epochs 30 \
 --pretrained_model_name_or_path ../../../pretrained_stable_diffusion/stable-diffusion-v1-5/v1-5-pruned.safetensors \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --network_weights ${network_weights} \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --anomal_only_on_object \
 --anomal_source_path "${anomal_source_path}" \
 --anomal_min_perlin_scale 0 \
 --anomal_max_perlin_scale 6 \
 --anomal_min_beta_scale 0.5 \
 --anomal_max_beta_scale 0.8 \
 --back_min_perlin_scale 0 \
 --back_max_perlin_scale 6 \
 --back_min_beta_scale 0.6 \
 --back_max_beta_scale 0.9 \
 --back_trg_beta 0 \
 --do_anomal_sample --do_object_detection --do_background_masked_sample --global_net_normal_training \
 --local_use_position_embedder --use_position_embedder \
 --trg_layer_list "['mid_block_attentions_0_transformer_blocks_0_attn2',
                    'up_blocks_1_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_2_attentions_2_transformer_blocks_0_attn2',
                    'up_blocks_3_attentions_2_transformer_blocks_0_attn2',]"