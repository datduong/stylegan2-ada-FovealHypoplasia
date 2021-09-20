

import sys,os,re,pickle
from PIL import Image
import numpy as np 


path = sys.argv[1]
folder_prefix = sys.argv[2]
interval = sys.argv[3]
try: 
  seed_range = [int(i) for i in sys.argv[4].split(',')]
except: 
  seed_range = [0,100]


outdir = os.path.join(path,folder_prefix)
if not os.path.exists(outdir): 
  os.makedirs(outdir)

mix_ratio = interval.strip().split() 
print (mix_ratio)

for seed in np.arange(seed_range[0],seed_range[1]): 
  image_list = []
  image_list = image_list + [os.path.join(path,folder_prefix+'M'+str(m),f'seed{seed:08d}.png') for m in mix_ratio ]
  images = [Image.open(x) for x in image_list]
  widths, heights = zip(*(i.size for i in images))
  total_width = sum(widths)
  max_height = max(heights)
  new_im = Image.new('RGB', (total_width, max_height))
  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]
  #
  new_im.save(os.path.join(outdir,f'seed{seed:04d}.png'))


