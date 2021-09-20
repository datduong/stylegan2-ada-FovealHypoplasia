

import os,sys,re,pickle
import pandas as pd 
import numpy as np

metricfile = 'metric-fid350_full.txt'
mainpath = '/data/duongdb/FH_OCT_08172021/Stylegan2'

these_runs = os.listdir(mainpath)

just_these = sorted ( [i.strip() for i in """00008-Tf256RmFold1+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00006-Tf256RmFold3+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00007-Tf256RmFold2+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00005-Tf256RmFold0+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00009-Tf256RmFold4+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3""".split()] )

these_runs = [i for i in just_these]
these_runs

for model in these_runs: 
  model = os.path.join(mainpath,model)
  os.chdir(model)
  if metricfile in os.listdir(model): 
    df = pd.read_csv(metricfile, sep='fid350_full')
    df.columns = ['0','1']
    names = df['0'].values
    print ( os.path.join(model,names [ np.argmin( [float (i) for i in df['1'].values] ) ] ) ) 
    print ( np.min( [float (i) for i in df['1'].values] ))
    print ('\n')

