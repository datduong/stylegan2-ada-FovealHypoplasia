import os,sys,re
import pickle

sys.path.append('/data/duongdb/stylegan2-ada')
import dnnlib
import dnnlib.tflib as tflib

import gzip
import json

import numpy as np
from tqdm import tqdm_notebook

import warnings
import matplotlib.pylab as plt

import numpy as np 
import pandas as pd 

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import Ridge

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score

import PIL.Image


def ArrayFromString(strinput): 
  strinput = re.sub(r'\[','',strinput)
  strinput = re.sub(r'\]','',strinput)
  strinput = re.sub(r'\n','',strinput).strip().split()
  return [np.round(float(s),2) for s in strinput]


def ExpectedAge(age): 
    return np.round ( np.sum( age * np.array([1,6,15,25,35,45,55,65,75]) ) , 4 ) # expected age

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dlatents_np", type=str, default=None)
parser.add_argument("--label_file", type=str, default=None)
parser.add_argument("--network_pkl", type=str, default=None)
parser.add_argument("--truncation_psi", type=float, default=None)
parser.add_argument("--colname", type=str, default=None)
parser.add_argument("--num_ws", type=int, default=14)
parser.add_argument("--style", type=str, default='0-6')
parser.add_argument("--range", type=float, default='5')
parser.add_argument("--dlatents_np_real", type=str, default=None)
parser.add_argument("--maxlen", type=float, default=3)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--losstype", type=str, default='log')

args = parser.parse_args()

args.style = [int(i.strip()) for i in args.style.split('-')]
args.style = np.arange(args.style[0],args.style[1])

args.range = list (np.arange(args.range*-1, args.range, step=args.range/args.maxlen)) + [args.range] + [0] # middle point is needed
args.range = sorted(set ( args.range) ) 
print ( 'range {}'.format(len (args.range) ) )

# ! simple logistic regression 

# ! do simple female/male ? why not. 

# dlatents_np = '/data/duongdb/WSq22_04202021/Model/00001-ImgTf256-paper256-ada-target0.7-resumeffhq256/allfakeimagesFromW/00002-project-real-images/FakeImgW14.512.pickle'
# num_ws = 14 

dlatent_data = pickle.load(open(args.dlatents_np,'rb')) # ! sorted by name
X_data = dlatent_data.reshape((-1, args.num_ws*512)) # ! reshape so it's batch x 14*512

# create label in numpy format. 
label_file = pd.read_csv(args.label_file) # ! make sure files are sorted, it should be if we call FairFace with "sorted"
if args.colname == 'gender':
    temp_ = list(label_file[args.colname])
    Y_logit = np.array([1 if i == 'Female' else 0 for i in temp_])

if args.colname == 'africa':
    temp_ = list(label_file['race_scores_fair'])
    Y_logit = []
    for t in temp_ : 
        Y_logit.append( np.round ( ArrayFromString(t)[1] > .25 ) ) # ! 2nd entry 
    Y_logit = np.array(Y_logit)
    # Y_logit = np.array([1 if i == 'Black' else 0 for i in temp_])
    
if args.colname == 'notwhite':
    temp_ = list(label_file['race'])
    Y_logit = np.array([1 if i != 'White' else 0 for i in temp_])

if args.colname == 'age': 
    temp_ = list(label_file['age_scores_fair'])
    Y_logit = []
    for t in temp_: 
        Y_logit.append(ExpectedAge(ArrayFromString(t)))
    Y_logit = np.array(Y_logit)
    # ! linear regression, not logistic on age
    clf = Ridge(alpha=1.0).fit(X_data,Y_logit)
    clf.fit(X_data, Y_logit)
    fitted_coef = clf.coef_.reshape((args.num_ws, 512))

# 
print (Y_logit)
# Y_logit = np.random.randint(2, size=X_data.shape[0]) # debug


# ! set up model and fit. 

# clf = LogisticRegression(max_iter=10000,class_weight='balanced').fit(X_data, Y_logit) # ! simple logistics
# fitted_coef = clf.coef_.reshape((args.num_ws, 512))
# print (fitted_coef)
# score = clf.score(X_data, Y_logit)
# print('logit score {}'.format(score))

if args.colname != 'age': 
    clf = SGDClassifier(args.losstype, n_iter_no_change=1000, max_iter=10000, class_weight='balanced') # SGB model for performance sake
    clf.fit(X_data, Y_logit)
    fitted_coef = clf.coef_.reshape((args.num_ws, 512))

# output    
print (fitted_coef)
score = clf.score(X_data, Y_logit)
print('logit score {}'.format(score))


# scores = cross_val_score(clf, X_data, Y_logit, scoring='accuracy', cv=5)
# print(scores)
# print('Mean: ', np.mean(scores))


# ! load generator, fit new images

# network_pkl='/data/duongdb/WSq22_04202021/Model/00001-ImgTf256-paper256-ada-target0.7-resumeffhq256/network-snapshot-002406.pkl'

tflib.init_tf()
print('Loading networks from "%s"...' % args.network_pkl)
with dnnlib.util.open_url(args.network_pkl) as fp:
    _G, _D, Gs = pickle.load(fp)

# Render images for dlatents initialized from random seeds.
w_avg = Gs.get_var('dlatent_avg') # [component] # ! for truncation 
Gs_syn_kwargs = {
                'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                'randomize_noise': False
                }

# truncation_psi = .6

def generate_image(latent_vector, truncation_psi): 
    # follow same code to generate images in "style-mixing"
    latent_vector = latent_vector.reshape((1, args.num_ws, 512)) # ! the W vec
    latent_vector = w_avg + (latent_vector - w_avg) * truncation_psi # [minibatch, layer, component]
    images = Gs.components.synthesis.run(latent_vector, **Gs_syn_kwargs)          
    return images[0] # return image in np format


def get_concat_h(im1, im2): # concat 2 images into 1 picture
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def move_and_show(latent_vector, direction, coeffs, foutname):
    # ! https://github.com/pbaylies/stylegan-encoder/blob/master/Learn_direction_in_latent_space.ipynb
    fig,ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        # ! @new_latent_vector[:7] so, we swap the first 7 layers, keep the other 7 layers unchanged
        new_latent_vector[args.style] = (latent_vector + coeff*direction)[args.style] # ! only swap some layers, they do 0:8
        if coeff == 0: 
            this_trunc = 1
        else: 
            this_trunc = args.truncation_psi # don't truncate if original is given. 
        #
        im = PIL.Image.fromarray ( generate_image(new_latent_vector, truncation_psi=this_trunc) ) 
        if i == 0: 
            frame = im # 1st image
        else: 
            frame = get_concat_h(frame, im)
            
        # ax[i].imshow(img)
        # ax[i].set_title('Coeff: %0.1f' % coeff)

    # ! save
    frame.save(foutname)
    # [x.axis('off') for x in ax]
    # plt.show()
    # plt.savefig(foutname+'.png')


# ! generate on fake images, just to debug
# for i in range(5):
#     move_and_show(X_data.reshape((-1, args.num_ws, 512))[i], fitted_coef, args.range, str(i)+'.png')

# ! generate on real images? 
if args.dlatents_np_real is None: exit() 

if not os.path.exists(args.outdir): os.mkdir(args.outdir)

dlatent_data = pickle.load(open(args.dlatents_np_real,'rb')) # ! real images 
X_data = dlatent_data.reshape((-1, args.num_ws*512)) # ! reshape so it's batch x 14*512
X_reshape = X_data.reshape((-1, args.num_ws, 512))

for i in range(X_data.shape[0]): # ! all the real images 
    move_and_show(X_reshape[i], fitted_coef, args.range, os.path.join(args.outdir,str(i)+'.png'))

