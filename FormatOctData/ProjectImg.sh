#!/bin/bash

# sinteractive --time=3:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

# ! align images into ffhq format # this has to be done so we can greatly leverage transfer-ability of ffhq
# resolution=256
# datapath=/data/duongdb/FH_OCT_08172021
# cd $datadir/stylegan2-ada/WsData # ! use styleflow... this may not work ??
# python3 AlignImage.py $datapath/SampleImg $datapath/AlignSampleImg$resolution $resolution

#----------------------------------------------------------------------------
# ! project on real images? 

model=/data/duongdb/FH_OCT_08172021/Model/00007-OverSample10.60kFF.Gender.Tf512InTrainsetFold1-paper512-kimg50000-ada-target0.6-resumeffhq512/network-snapshot-001152.pkl

cd /data/duongdb/stylegan2-ada

truncationpsi=0.6 # @trunc=0.7 is recommended on their face dataset. 

# ! convert into tfrecords

# cd /data/duongdb/stylegan2-ada

# maindir=/data/duongdb/FH_OCT_08172021

# for imgtype in 22q11DSearly WSearly 22q11DSlate WSlate # WSearly
# do 
#   cd /data/duongdb/stylegan2-ada
#   outdir=$maindir/aligned_images_$imgtype'_Tf'
#   rm -rf $outdir
#   mkdir $outdir
#   python3 dataset_tool.py create_from_images $outdir $maindir/aligned_images_$imgtype --resolution 256 --shuffle 0 # we don't make labels, because we just go from W-->Img
# done 

# # ! project on real images

# cd /data/duongdb/stylegan2-ada 
# maindir=/data/duongdb/FH_OCT_08172021

# tot_aligned_imgs=15 # ! change this number
# truncationpsi=0.5 # @trunc=0.7 is recommended on their face dataset. 

# # ! project on real images
# for imgtype in 22q11DSearly # WSearly 22q11DSlate WSlate # late inter WSearly 22q11DSearly
# do 
#   if [ $imgtype == '22q11DSearly' ]; then
#     classid=0
#   fi
#   if [ $imgtype == '22q11DSlate' ]; then
#     classid=2
#   fi
#   if [ $imgtype == 'WSearly' ]; then
#     classid=3
#   fi
#   if [ $imgtype == 'WSlate' ]; then
#     classid=5
#   fi
#   python3 run_projector.py project-real-images --network=$model --data-dir=$maindir/aligned_images_$imgtype'_Tf' --num-images=$tot_aligned_imgs --classid $classid --result-dir $maindir/aligned_images_$imgtype'_proj_notile' --tile 0 --num_steps 10000 
# done


# ! mix the projected W

# ! where is dlatent (this is the W)
cd /data/duongdb/stylegan2-ada 
maindir=/data/duongdb/FH_OCT_08172021
# C:\Users\duongdb\Documents\FH_OCT_08172021\aligned_images_22q11DSearly_proj_notile\00000-project-real-images
outdir=$maindir/Classify/ProjImgTf60kDlatent/F1L22q11DS_early_test_trial
mkdir $outdir
dlatents_np=$maindir/Classify/ProjImgTf60kDlatent/F1L22q11DS_early
python3 style_mixing.py --network $model --outdir $outdir --trunc .5 --num_labels 6 --dlatents_np $dlatents_np --rows 1,2,5,0,6,7 --cols 3,4 --styles 0-4 --save_individual 1


outdir=$maindir/aligned_images_WSearly_proj_notile/00001-project-real-images
mkdir $outdir
dlatents_np=$maindir/aligned_images_WSearly_proj_notile/00001-project-real-images
python3 style_mixing.py --network $model --outdir $outdir --trunc .5 --num_labels 6 --dlatents_np $dlatents_np --rows 3,4,5,0,1,2 --cols 6,7 --styles 0-3


# ! mix on fake images
cd /data/duongdb/stylegan2-ada 
maindir=/data/duongdb/FH_OCT_08172021
outdir=$maindir/MixExample
mkdir $outdir
python3 style_mixing.py --network $model --outdir $outdir --trunc .6 --num_labels 6 --class 5 --rows 123,100,55,42,1000 --cols 2000,300 --styles 0-3




