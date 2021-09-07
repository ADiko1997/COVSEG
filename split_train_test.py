import shutil
import os
import random
import numpy as np

np.random.seed(0)
random.seed(0)



home = os.getcwd()
N_patient = os.listdir(home + "/ct_lesion_seg/mask")
index =0
for i in range(len(N_patient)):
    if N_patient[i] == ".DS_Store":
      # print(i)
      index = i
 
N_patient.pop(index)
 
# for i in range(len(N_patient)):
#     print(N_patient[i],"index ",i)          #creo lista con file dentro$
#del N_patient[0]                                      #rimuovo Ds_store dala lista
random.shuffle(N_patient)                             #dispone in modo casuale i pazienti$
  
    
newpath = home + '/test'
if not os.path.exists(newpath):                     #crea la cartella test
    os.mkdir(newpath)
 
newpath = home + '/test_labels'
if not os.path.exists(newpath):
    os.mkdir(newpath)
    
newpath = home + '/train'
if not os.path.exists(newpath):                     #crea la cartella train
    os.mkdir(newpath)
 
newpath = home + '/train_labels'
if not os.path.exists(newpath):
    os.mkdir(newpath)
   
base_dir = home + '/ct_lesion_seg/mask'                   #definisco due variabili base_d$
base_dir1 = home + '/ct_lesion_seg/image'
 
 
 
for i in range(0,120):
    N_slice=os.listdir(home+"/ct_lesion_seg/mask/"+N_patient[i])              
    for j in range(0,len(N_slice)):
        Name=N_patient[i]+'_'+N_slice[j]
 
        in_name=os.path.join(base_dir,N_patient[i],N_slice[j])
        out_name=os.path.join(base_dir,N_patient[i],Name)             
        os.rename(in_name,out_name)
 
        in_name1=os.path.join(base_dir1,N_patient[i],N_slice[j][:-4]+'.jpg')
        out_name1=os.path.join(base_dir1,N_patient[i],Name[:-4]+'.jpg')           
        os.rename(in_name1,out_name1)
 
        shutil.move(out_name,home+'/train_labels')                                   
        shutil.move(out_name1,home+'/train')
 
for i in range(120,150):
    N_slice=os.listdir(home + "/ct_lesion_seg/mask/" + N_patient[i])             
    for j in range(0,len(N_slice)):
        Name=N_patient[i]+'_'+N_slice[j]
 
        in_name=os.path.join(base_dir,N_patient[i],N_slice[j])
        out_name=os.path.join(base_dir,N_patient[i],Name)
        os.rename(in_name,out_name)
 
        in_name1=os.path.join(base_dir1,N_patient[i],N_slice[j][:-4]+'.jpg')
        out_name1=os.path.join(base_dir1,N_patient[i],Name[:-4]+'.jpg')
        os.rename(in_name1,out_name1)
 
        shutil.move(out_name,home + '/test_labels') 
        shutil.move(out_name1,home + '/test')