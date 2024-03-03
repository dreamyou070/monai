# !/bin/bash
# scratch vae / trained vae
# matching losses : only normal

port_number=50005
bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
file_name="monai_trial_ddpm"

anomal_source_path="../../../MyData/anomal_source"

# having anomal image only on object

python ../anomal_ddpm_conditioned_train.py \
 --output_dir "../../result/${bench_mark}/${obj_name}/${file_name}" \
 --start_epoch 0 --max_train_epochs 100 \
 --data_path "../../../MyData/anomaly_detection/${bench_mark}" \
 --trigger_word "${trigger_word}" \
 --obj_name "${obj_name}" --anomal_only_on_object \
 --anomal_source_path "/home/dreamyou070/MyData/anomal_source" \
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