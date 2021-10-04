import re,sys,os,pickle

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12
# sinteractive --time=1:00:00 --gres=gpu:v100x:1 --mem=20g --cpus-per-task=32 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:v100x:2 --mem=24g --cpus-per-task=24 

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37

module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! set up path, resolution etc...

outputdir=/data/duongdb/FH_OCT_08172021

resolution=256

# ! tf dataset 

imagedata=$outputdir/Classify/Tf256RmFoldFOLD+EyePos+FH 

# ! resume ? 

resume=ffhq$resolution

# ! train 

cd /data/duongdb/stylegan2-ada-EyeOct
python3 train_with_labels.py \
--data=$imagedata \
--gpus=2 --target=0.8 \
--aug=ada \
--outdir=$outputdir/Stylegan2 \
--resume=$resume \
--cfg=paper$resolution \
--snap=10 \
--oversample_prob=0 \
--mix_labels=0 \
--metrics=fid350_full \
--kimg 3000 \
--split_label_emb_at 4


"""
import time
from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

os.chdir('/data/duongdb/stylegan2-ada-EyeOct')
counter = 1

for fold in [0,1,2,3,4]: 
  newscript = re.sub('FOLD',str(fold),script)
  fname = 'script'+str(counter+1)+date_time+'.sh'
  fout = open(fname,'w')
  fout.write(newscript)
  fout.close()
  counter = counter + 1
  time.sleep(2)
  # os.system('sbatch --partition=gpu --time=6:00:00 --gres=gpu:p100:2 --mem=16g --cpus-per-task=20 '+fname)



