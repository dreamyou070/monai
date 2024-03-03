# !/bin/bash
# scratch vae / trained vae
# matching losses : only normal

bench_mark="MVTec"
obj_name='transistor'
trigger_word='transistor'
file_name="monai_trial_ddpm"

anomal_source_path="../../../MyData/anomal_source"

python ../anomal_ddpm_classifier_guidance_infer.py \
 --output_dir "../../result/${bench_mark}/${obj_name}/${file_name}" \