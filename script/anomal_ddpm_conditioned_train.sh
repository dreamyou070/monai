# !/bin/bash

port_number=50033
bench_mark="Tuft"
obj_name='teeth_20240308'
trigger_word='teeth'
file_name="monai_trial_teeth"

anomal_source_path="../../../MyData/noise_source"

# having anomal image only on object

accelerate launch --config_file ../../../gpu_config/gpu_0_config \
 --main_process_port $port_number ../anomal_ddpm_conditioned_train.py \
 --output_dir "../../result/${bench_mark}/${obj_name}/${file_name}" \
 --start_epoch 0 --max_train_epochs 100 \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --anomal_only_on_object \
 --anomal_source_path "${anomal_source_path}" \
 --min_perlin_scale 1 \
 --max_perlin_scale 4 \
 --max_beta_scale 0.9 \
 --min_beta_scale 0.5 \
 --do_self_aug