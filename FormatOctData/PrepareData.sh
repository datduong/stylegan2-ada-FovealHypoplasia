#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 
# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=4

# ! make tfrecord data. with json label, REMOVE TEST FOLD 0

cd /data/duongdb/stylegan2-ada-EyeOct

# classifier_train_csv='/data/duongdb/FH_OCT_08172021/FH_OCTs_label_train_input.csv' # ! fh score 1,2,3,4
# jsonname=dataset_eyepos_FH.json

classifier_train_csv='/data/duongdb/FH_OCT_08172021/FH_OCTs_label_train_input_driving.csv' # ! driving label class A,B,C
jsonname=dataset_eyepos_driving.json

datapath=/data/duongdb/FH_OCT_08172021
for fold in 0 1 2 3 4
do 
  for resolution in 256  
  do 
  # outdir=$datapath/Classify/Tf$resolution'RmFold'$fold'+EyePos+FH' # ! FH score 1,2,3,4
  outdir=$datapath/Classify/Tf$resolution'RmFold'$fold'+EyePos+Driving' # ! Driving A,B,C
  python3 dataset_tool.py create_from_images_with_labels_fromjson $outdir $datapath/FH_OCT_Images --resolution $resolution --labeljson_name $jsonname --classifier_train_csv $classifier_train_csv --fold $fold
  done 
done 

