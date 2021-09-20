#!/bin/bash

# sinteractive --time=12:00:00 --gres=gpu:k80:1 --mem=12g --cpus-per-task=12
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2:00:00 --gres=gpu:k80:1 --mem=12g --cpus-per-task=12 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 


source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

cd /data/duongdb/stylegan2-ada

maindir='/data/duongdb/WSq22_04202021/Model'
modeldir=$maindir/'00001-ImgTf256-paper256-ada-target0.7-resumeffhq256'

network_pkl=$modeldir/network-snapshot-002406.pkl
dlatents_np=$modeldir/ProjFakeImg/FakeImgW14.512.pickle
label_file=$modeldir/ProjFakeImg/FairFaceOutput.csv
truncation_psi=0.9 # ! possible to use larger psi? 

dlatents_np_real=$modeldir/ProjFakeImg/RealImgW14.512.pickle

# colname='africa'

# '0-6' '0-5' '0-4' '0-3' '0-2' '0-1' '0-7' '0-8' '0-9' '0-10' '0-11' '0-12' '0-13' '0-14'

for colname in age # africa notwhite gender
do
  for style in '0-6' '0-5' '0-4' '0-3' '0-2' '0-1' '0-7' '0-8' '0-9' '0-10' '0-11' '0-12' '0-13' '0-14'
  do 
  # ! age--> need larger range, tried 40, looked ok
  outdir=$modeldir/$colname'T'$truncation_psi'S'$style
  python3 AuxClassifier/LogitRegression.py --dlatents_np $dlatents_np --label_file $label_file --network_pkl $network_pkl --truncation_psi $truncation_psi --colname $colname --num_ws 14 --style $style --dlatents_np_real $dlatents_np_real --range 4 --maxlen 4 --outdir $outdir --losstype 'log'
  done 
done 

