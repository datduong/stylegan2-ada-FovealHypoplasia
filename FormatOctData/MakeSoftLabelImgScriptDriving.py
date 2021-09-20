import re,sys,os,pickle

# sinteractive --time=2:00:00 --gres=gpu:p100:1 --mem=12g --cpus-per-task=12
# sbatch --partition=gpu --time=2-00:00:00 --gres=gpu:p100:2 --mem=24g --cpus-per-task=24 

script="""#!/bin/bash

source /data/$USER/conda/etc/profile.d/conda.sh
conda activate py37
module load CUDA/11.0
module load cuDNN/8.0.3/CUDA-11.0
module load gcc/8.3.0

fold=FOLD 

if [ $fold == 0 ]; then
  # ! model path
  model=/data/duongdb/FH_OCT_08172021/Stylegan2/00005-Tf256RmFold0+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3/network-snapshot-000614.pkl
fi

if [ $fold == 1 ]; then
  model=/data/duongdb/FH_OCT_08172021/Stylegan2/00008-Tf256RmFold1+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3/network-snapshot-000614.pkl
fi

if [ $fold == 2 ]; then
  model=/data/duongdb/FH_OCT_08172021/Stylegan2/00007-Tf256RmFold2+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3/network-snapshot-000563.pkl
fi

if [ $fold == 3 ]; then
  model=/data/duongdb/FH_OCT_08172021/Stylegan2/00006-Tf256RmFold3+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3/network-snapshot-000614.pkl
fi

if [ $fold == 4 ]; then
  model=/data/duongdb/FH_OCT_08172021/Stylegan2/00009-Tf256RmFold4+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3/network-snapshot-000512.pkl
fi


truncationpsi=TRUNCATION # @trunc

cd /data/duongdb/stylegan2-ada-EyeOct 

outdir=OUTPUTDIR
mkdir $outdir

class='CLASS'
classnext=NEXT 

label=LABEL

for mix in MIXRATIO
do

  python3 generate.py --outdir=$outdir --trunc=$truncationpsi --seeds=SEED --network $model --savew 0 --suffix 'F'$fold'C'$class'C'$classnext'M'$mix'T'$truncationpsi$label --mix_ratio $mix --class=$class --class_next=$classnext

done 


"""

import time
from datetime import datetime
import numpy as np 
import pandas as pd 

now = datetime.now() # current date and time
date_time = now.strftime("%m%d%Y%H%M%S")

os.chdir('/data/duongdb/stylegan2-ada-EyeOct')

# ------------------------------------------------------------------------------------------

name_option = '' # ! can put in your own name
TRAINCSV = '/data/duongdb/FH_OCT_08172021/FH_OCTs_label_train_input_driving.csv'
TRAINCSV = pd.read_csv ( TRAINCSV, dtype=str ) 

MIXRATIO = .9
TRUNCATION = .8

# ------------------------------------------------------------------------------------------


labelset_head = sorted('A,B,C'.split(','))
labelset_head_in_gan = {v.strip():i for i,v in enumerate(labelset_head)}

labelset_tail = sorted('OD,OS'.split(','))
labelset_tail_in_gan = {v.strip():(i+len(labelset_head)) for i,v in enumerate(labelset_tail)} # ! shift +3 because label takes the first 3 spots

def make_label_vec (thisname,labelset_head_in_gan,labelset_tail_in_gan): 
  vec = []
  for head in labelset_head_in_gan: 
    if head in thisname: 
      vec.append ( labelset_head_in_gan[head] ) 
      break 
  for tail in labelset_tail_in_gan: 
    if tail in thisname: 
      vec.append( labelset_tail_in_gan[tail] )
      break 
  print ('label is {} vector is {}'.format(thisname,vec))
  return ','.join(str(v) for v in vec)


import itertools
def make_label_pairs (name_array1,name_array2,labelset_head_in_gan,labelset_tail_in_gan): 
  pairs = { }
  for n1 in name_array1: # condition on "fixed", this is "eye" because eyes don't change 
    for n2 in list(itertools.combinations(name_array2, 2)): # @n2 is array of tuple [(disease1,disease2)...]
      if ('A' in n2 and 'B' in n2) or ('B' in n2 and 'C' in n2) : 
        t1 = make_label_vec(n2[0]+n1, labelset_head_in_gan,labelset_tail_in_gan)
        t2 = make_label_vec(n2[1]+n1, labelset_head_in_gan,labelset_tail_in_gan)
        pairs[n2[0]+n1+'_'+n2[1]+n1] = [t1,t2]
        pairs[n2[1]+n1+'_'+n2[0]+n1] = [t2,t1]
  return (pairs)

#
label_pair = make_label_pairs(labelset_tail, labelset_head,labelset_head_in_gan,labelset_tail_in_gan)
print (label_pair)


temp = []
for i in labelset_head: 
  for j in labelset_tail: 
    temp.append(i+j)


#
labelset=sorted(temp)
label_seed = {}
for i,l in enumerate(labelset): 
  label_seed[l] = i * 500

# ------------------------------------------------------------------------------------------

#
rootout = '/data/duongdb/FH_OCT_08172021/Classify/'+name_option+'Soft+Driving+Eye'+'M'+str(MIXRATIO)+'T'+str(TRUNCATION)
if not os.path.exists(rootout): 
  os.mkdir(rootout)

# 

MULTIPLY_BY = 1 # ! how many times we do style mix ?

fold_seed = {i : i*10000 for i in range(5)}

label_pair_key = sorted(list(label_pair.keys()))

counter = 1
for fold in [0,1,2,3,4]: # ,2,3,4,5
  print ('fold {}'.format(fold))
  OUTPUTDIR=os.path.join(rootout,'F'+str(fold)+'X'+str(MULTIPLY_BY))
  if not os.path.exists (OUTPUTDIR):  
    os.mkdir(OUTPUTDIR)
  for selectlabel in label_pair_key: 
    print (selectlabel)
    label1 = selectlabel.split('_')[0]
    if ('Normal' in label1): # or ('B' in label1): 
      continue # ! dont need to make more normal
    label2 = selectlabel.split('_')[1]
    if label1 == label2: 
      continue # ! skip same label, should not happen anyway
    #
    class1 = label_pair[selectlabel][0]
    class2 = label_pair[selectlabel][1]
    #
    classifier_train_csv = TRAINCSV[TRAINCSV['fold'] != fold ] 
    label1_temp = label1[0] # ! remove left/right from label, take 1st character
    classifier_train_csv = classifier_train_csv[classifier_train_csv['label'] == label1_temp ] # ! keep labels needed
    #
    if label1_temp in ['A','C']: 
      imglist = np.arange ( classifier_train_csv.shape[0] ) # ! don't need this "//2" ?? we don't have A->B A->C, so we double, to keep same size, we need //2
    else: 
      imglist = np.arange ( classifier_train_csv.shape[0]//4 ) # ! don't need this "//2" ?? we don't have A->B A->C, so we double, to keep same size, we need //2
    print ('img size for this pair {}'.format(len(imglist)))
    # ! make seed, so A-B, and A-C must have same seed which is based on A
    seed_start = fold_seed[fold] + label_seed[label1] # ! off set the label seed by the fold seed
    seed_end = seed_start + len(imglist)
    #
    newscript = re.sub('MIXRATIO',str(MIXRATIO),script)
    newscript = re.sub('TRUNCATION',str(TRUNCATION),newscript)
    newscript = re.sub('FOLD',str(fold),newscript)
    newscript = re.sub('CLASS',str(class1),newscript)
    #
    newscript = re.sub('NEXT',str(class2),newscript)
    #
    newscript = re.sub('LABEL',label1,newscript)
    newscript = re.sub('OUTPUTDIR',OUTPUTDIR,newscript)
    newscript = re.sub('SEED',str(seed_start)+'-'+str(seed_end),newscript) # ! //2 half female/male
    fname = 'script'+str(counter+1)+date_time+'.sh'
    fout = open(fname,'w')
    fout.write(newscript)
    fout.close()
    counter = counter + 1
    time.sleep(1)
    # break
    os.system('sbatch --partition=gpu --time=00:20:00 --gres=gpu:k80:1 --mem=6g --cpus-per-task=4 '+fname)




