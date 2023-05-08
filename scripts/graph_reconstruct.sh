#!/bin/bash

cuda=$1

database_name="graph_reconstruction"
for dataset_name in DD # AIDS ENZYMES NCI1 PROTEINS
do
  for target_model in mean_pool mincut_pool diff_pool
    do
      python main.py --is_vary false --exp 'graph_recon' --cuda $cuda \
      --dataset_name $dataset_name --shadow_dataset $dataset_name \
      --target_model $target_model --encoder_method $target_model \
      --is_gen_recon_data true --is_use_feat true \
      --is_upload true --database_name $database_name --num_runs 3 \
      2>&1 | tee "./temp_data/log/$dataset_name.$target_model.reconstruct.log" &
    done
done
