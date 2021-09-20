

# ! delete runs with high FID score 

import os,sys,re,pickle 
import json 

rootpath = '/data/duongdb/FH_OCT_08172021/'

thesefolders = [i for i in os.listdir(rootpath) if 'Stylegan2' in i]

just_these = [i.strip() for i in """00011-Tf256RmTestFold1+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00012-Tf256RmTestFold2+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00013-Tf256RmTestFold3+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00014-Tf256RmTestFold4+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3
00015-Tf256RmTestFold5+EyePos+Driving-paper256-kimg3000-ada-target0.8-resumeffhq256-divlabel3""".split()]

for thesefolder in thesefolders:
  folderpath = os.path.join(rootpath,thesefolder) 
  for folder in os.listdir(folderpath): 
    if folder not in just_these: 
      continue
    os.chdir(os.path.join(folderpath,folder))
    # 
    os.system ( 'rm ' + os.path.join(folderpath,folder,'fakes_init.png') )
    # reading the data from the file 
    try: 
      fin = open ('metric-fid350_full.txt','r')
    except: 
      continue
    #
    print (folder)     
    for d in fin: 
      if len(d) > 0: 
        thisline = d.split()
        value = float ( thisline[-1] ) 
        if value > 51: 
          todelete = thisline[0]+'.pkl'
          print (todelete) # ! may see error if files are deleted already
          if os.path.exists(os.path.join(folderpath,folder,todelete)): 
            os.system ( 'rm ' + os.path.join(folderpath,folder,todelete)) # 'network-snapshot-000000.pkl'
            idcode = todelete.split('-')[-1]
            idcode = re.sub('.pkl','',idcode)
            print (idcode)
            os.system ( 'rm ' + os.path.join(folderpath,folder,'fakes' + idcode + '.png') )




