
import os,sys,re,pickle
import numpy as np 
import pandas as pd 

# ! need to gather all the Ws together. 

foutpath = '/data/duongdb/WSq22_04202021/Model/00001-ImgTf256-paper256-ada-target0.7-resumeffhq256/ProjFakeImg'

all_w = []

# dlatents_np_array=['/data/duongdb/WSq22_04202021/Model/00001-ImgTf256-paper256-ada-target0.7-resumeffhq256/ProjFakeImg/00000-project-real-images',
#                    '/data/duongdb/WSq22_04202021/Model/00001-ImgTf256-paper256-ada-target0.7-resumeffhq256/ProjFakeImg/00001-project-real-images'
#                    ]

dlatents_np_array=['/data/duongdb/WSq22_04202021/aligned_images_22q11DSearly_proj_notile/00001-project-real-images',
                   '/data/duongdb/WSq22_04202021/aligned_images_WSearly_proj_notile/00001-project-real-images'
                   ] 

for dlatents_np in dlatents_np_array : 
  #  
  # ! python sort is not true natural order sorting
  npz = sorted ( [f for f in os.listdir(dlatents_np) if 'npz' in f] ) # ! make life easy by sorting. 
  print (len(npz))
  #
  for f in npz: 
    w = np.load(os.path.join(dlatents_np,f))['dlatents']
    all_w.append(w) 
  

# 
all_w = np.concatenate(all_w) # ! (*, 14, 512)
print (all_w.shape)
pickle.dump(all_w,open(os.path.join(foutpath,'RealImgW14.512.pickle'), 'wb'))

