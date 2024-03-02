# !/bin/bash
# scratch vae / trained vae
# matching losses : only normal

port_number=50005
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
layer_name='layer_3'
sub_folder="up_16_32_64"
file_name="monai_trial"

anomal_source_path="../../../MyData/anomal_source"

python ../anomal_ddpm_classifier_guidance.py \
 --output_dir "../../result/${bench_mark}/${obj_name}/${layer_name}/${sub_folder}/${file_name}" \
 --start_epoch 0 --max_train_epochs 30 \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
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
 --do_anomal_sample --do_object_detection