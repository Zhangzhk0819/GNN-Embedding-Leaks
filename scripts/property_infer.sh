#!/bin/bash
# run.sh
dataset=$1
cuda=$2
for property_num_class in 2 4 6 8
do
	  for target_model in diff_pool mincut_pool mean_pool
	  do
		  printf $dataset $target_model $property_num_class
		  python main.py --attack 'property_infer' --dataset $dataset --property_num_class $property_num_class --target_model $target_model --cuda $cuda 2>&1 | tee "./temp_data/log/$dataset.$target_model.$property_num_class.log" &
	  done
done
wait
