
# ! make a json 

# "people": [{"name": "Scott", "website": "stackabuse.com", "from": "Nebraska"}, {"name": "Larry", "website": "google.com", "from": "Michigan"}, {

import os,sys,re,pickle
import json
import pandas as pd 
import numpy as np 

main_dir = '/data/duongdb/FH_OCT_08172021'
os.chdir(main_dir)

image_dir = os.path.join(main_dir,'FH_OCT_Images')


# traincsv = 'FH_OCTs_label_train_input.csv'
# testcsv = 'FH_OCTs_label_test_input.csv' # ! let's combine both train and test, we can filter out test images later
# outjson = 'dataset_eyepos_FH.json'

traincsv = 'FH_OCTs_label_train_input_driving.csv'
testcsv = 'FH_OCTs_label_test_input_match_train_col_driving.csv' # ! let's combine both train and test, we can filter out test images later
outjson = 'dataset_eyepos_driving.json'

label_json = {}
label_json['labels'] = []

label_file = pd.read_csv(traincsv, dtype=str)
label_file = pd.concat( [label_file, pd.read_csv(testcsv, dtype=str)], axis=0 ) 

# name,path,label,person_id,fold,eye_position_od,eye_position_os,machine_type_z,machine_type_hb,age_taken,logMAR,spherical_equivalent,nystagmus,dAchromatopsia,dAniridia,dCHS,dErdheim

labelset = sorted ( list ( set ( label_file['label'].values ) ) )
labelset = {val:i for i,val in enumerate(labelset)}

# ! set left/right eyes
eye_position = sorted(['OD','OS']) 

for index,row in label_file.iterrows():  
  condition = [0] * len(labelset) # replicate this
  condition[labelset[row['label']]] = 1 # 1hot
  #
  # ! add normal ? but we don't know what eye a normal is ? 
  if '_OD_' in row['name']: 
    eye = [1,0]
  else: 
    eye = [0,1]
  #
  label = condition + eye
  #
  label_json['labels'].append([row['name'],label]) # {img1:[0], img2:[1] ...}


#
with open(os.path.join(image_dir,outjson), 'w') as outfile:
  json.dump(label_json, outfile)

#
print (len(label_json['labels']))

