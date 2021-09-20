#!/bin/bash

# sinteractive --time=4:00:00 --gres=gpu:p100:1 --mem=8g --cpus-per-task=8
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

#----------------------------------------------------------------------------

# ! project on real images?
# ! first, we need to convert into tfrecords

maindir=/data/duongdb/FH_OCT_08172021

for imgtype in '1' '2' '3' '4'
do 
  cd /data/duongdb/stylegan2-ada-EyeOct
  outdir=$maindir/images_$imgtype'_Tf'
  rm -rf $outdir
  mkdir $outdir
  classifier_train_csv='/data/duongdb/FH_OCT_08172021/FH_OCTs_label_6fold.csv' # ! try with the large dataset
  fold=0 # ! must remove fold 0 too ?
  selectlabel=$imgtype
  # ! we don't make labels, because we just go from W-->Img
  python3 dataset_tool.py create_from_images $outdir $maindir/FH_OCT_Images --resolution 256 --shuffle 0 --classifier_train_csv $classifier_train_csv --fold 0,$fold  --selectlabel $selectlabel
done 


#----------------------------------------------------------------------------

# ! project on real images

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada-EyeOct 
maindir=/data/duongdb/FH_OCT_08172021

model=/data/duongdb/FH_OCT_08172021/Stylegan2/00000-Tf256RmTestFold0-paper256-kimg3000-ada-target0.8-resumeffhq256/network-snapshot-000845.pkl

tot_aligned_imgs=25 # ! change this number

truncationpsi=0.7 # @trunc=0.7 is recommended on their face dataset. 

# ! project on real images
for imgtype in '1' # WSearly 22q11DSlate WSlate # late inter WSearly 22q11DSearly
do 
  if [ $imgtype == '1' ]; then
    classid=0
  fi
  if [ $imgtype == '2' ]; then
    classid=1
  fi
  if [ $imgtype == '3' ]; then
    classid=2
  fi
  if [ $imgtype == '4' ]; then
    classid=3
  fi
  python3 run_projector.py project-real-images --network=$model --data-dir=$maindir/images_$imgtype'_Tf' --num-images=$tot_aligned_imgs --classid $classid --result-dir $maindir/images_$imgtype'_proj_notile' --tile 0 --num_steps 1000 --save_img 1
done


# ! mix the projected W on REAL IMAGES
# ! where is dlatent (this is the W)
cd /data/duongdb/stylegan2-ada-EyeOct 
maindir=/data/duongdb/FH_OCT_08172021

outdir=$maindir/images_1_proj_notile/StyleMix/
mkdir $outdir
dlatents_np=$maindir/images_1_proj_notile/00001-project-real-images
python3 style_mixing.py --network $model --outdir $outdir --trunc .7 --dlatents_np $dlatents_np --rows 1,2,6,14 --cols 3,4,8,9 --styles 0-4 
cd $outdir


