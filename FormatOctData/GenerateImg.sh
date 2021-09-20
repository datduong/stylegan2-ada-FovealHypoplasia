#!/bin/bash

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

#----------------------------------------------------------------------------

# ! generate images, using labels indexing
# ! let's try same random vector, but different label class

# @modelname takes on folder output from your stylegan2 experiment

for modelname in 00001-Tf256RmFold0+EyePos+FH-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel4/network-snapshot-000409.pkl 
do

  outdir=/data/duongdb/FH_OCT_08172021/Stylegan2/$modelname'Interpolate'
  mkdir $outdir

  model=/data/duongdb/FH_OCT_08172021/Stylegan2/$modelname

  cd /data/duongdb/stylegan2-ada-EyeOct

  truncationpsi=0.8 # @trunc=0.7 is recommended on their face dataset. 

  # ! generate
  class='0,4'
  class_next='0,5'
  for mix_ratio in 1 .8 .6 .4 .2 0
  do 
    python3 generate.py --outdir=$outdir/$class$class_next'T'$truncationpsi'M'$mix_ratio --trunc=$truncationpsi --seeds=0-50 --class=$class --class_next=$class_next --network $model --savew 0 --mix_ratio $mix_ratio
  done 

  # ! combine generated images into a single strip of images
  cd /data/duongdb/stylegan2-ada-EyeOct
  python3 concat_generated_img.py $outdir $class$class_next'T'$truncationpsi '1 .8 .6 .4 .2 0'
  echo $outdir/$class$class_next'T'$truncationpsi

done 
