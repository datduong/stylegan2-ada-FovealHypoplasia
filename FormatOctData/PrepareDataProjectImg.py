# cd /data/duongdb/stylegan2-ada/WS22qOther
# sbatch --time=12:00:00 --mem=24g --cpus-per-task=12 PrepareDataOverSample.sh

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

# ! we have to do this 5 folds, this is true replication 

classifier_train_csv=TRAINCSV

fold=FOLD

selectlabel=LABEL

datapath=/data/duongdb/WS22qOther_06012021/

datapathlist=$datapath/Align512 

name_option=NAMEOPTION

resolution=256 # ! 256 seems better than 512 because of small data 

outdir=$datapath/Classify/ProjImgTf$resolution$name_option
mkdir $outdir

outdir=$outdir/F$fold'L'$selectlabel

rm -rf $outdir # ! have to remove or we will see error. 

# ! we don't make labels, because we just go from W-->Img
cd /data/duongdb/stylegan2-ada
python3 dataset_tool.py create_from_images $outdir $datapathlist --resolution $resolution --shuffle 0 --classifier_train_csv $classifier_train_csv --fold $fold --selectlabel $selectlabel

"""

import os,sys,re,pickle
import time

from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

from numpy.lib.type_check import common_type 

# ------------------------------------------------------------------------------------------

name_option = 'AddNormalSplitSet' # ! let's try with almost equal sample sizes

TRAINCSV = '/data/duongdb/WS22qOther_06012021/Classify/train-oursplit'+name_option+'.csv'

# ------------------------------------------------------------------------------------------

labelset='WS_early,WS_late,WS_inter,22q11DS_early,22q11DS_late,22q11DS_inter'.split(',') # Controls_early,Controls_inter,Controls_late

labelset='22q11DS2y,22q11DSadolescence,22q11DSolderadult,22q11DSyoungadult,22q11DSyoungchild,Controls2y,Controlsadolescence,Controlsolderadult,Controlsyoungadult,Controlsyoungchild,WS2y,WSadolescence,WSolderadult,WSyoungadult,WSyoungchild'.split(',')

labelset={v:i for i,v in enumerate(sorted(labelset))}

os.chdir('/data/duongdb/stylegan2-ada')
counter = 0
for fold in [0,1,2,3,4]: 
  for label in labelset:
    counter = counter + 1
    newscript = re.sub('FOLD',str(fold),script)  
    newscript = re.sub('LABEL',str(label),newscript)  
    newscript = re.sub('TRAINCSV',TRAINCSV,newscript)  
    newscript = re.sub('NAMEOPTION',name_option,newscript)  
    fout = open('script'+str(counter)+date_time+'.sh','w')
    fout.write(newscript)
    fout.close() 
    time.sleep(2)
    os.system('sbatch --time=00:30:00 --mem=8g --cpus-per-task=4 ' + 'script'+str(counter)+date_time+'.sh')




