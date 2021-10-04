#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! set up path, resolution etc...

outputdir=/data/duongdb/FH_OCT_08172021

resolution=256

# ! tf dataset 

imagedata=$outputdir/Classify/Tf256RmFold3+EyePos+FH 

# ! resume ? 

resume=ffhq$resolution

# ! train 

cd /data/duongdb/stylegan2-ada-EyeOct
python3 train_with_labels.py --data=$imagedata --gpus=2 --target=0.8 --aug=ada --outdir=$outputdir/Stylegan2 --resume=$resume --cfg=paper$resolution --snap=10 --oversample_prob=0 --mix_labels=0 --metrics=fid350_full --kimg 3000 --split_label_emb_at 4


