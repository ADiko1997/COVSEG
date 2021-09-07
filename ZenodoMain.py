#!/usr/bin/env python3
# coding: utf-8

# In[1]:


import os
from PIL import Image
import numpy as np
import torch
import nibabel as nib
from itertools import chain
import matplotlib.pyplot as plt
import re
import shutil as sh

# read jpg in array (single slice)
#image=np.array(Image.open("87.jpg"))
#print(image.shape)

# read all
# 0 bkg 1 lung 2 GGO 3 CO
# forse l'indice ce ne e' uno in piu'

image_path = "/Users/cecilia/Documents/Documents_work/COVID/zenodo_vti/"

for i in os.listdir(image_path):
    
    OriginalVti ="/Users/cecilia/Documents/Documents_work/COVID/zenodo_vti/"+ i
    if OriginalVti.find(".vti") == -1:
        continue
    if OriginalVti.find("corona") == -1:
        continue
    if OriginalVti.find("mask") != -1:
        continue
    CaseNumber = i[12:-4]
    #temp CV
    if CaseNumber.find("008") == -1:
        continue
    SliceLDir = "/Users/cecilia/Documents/Documents_work/COVID/ZenodoNewPipe3/"+ CaseNumber + "S" # una volta tolto prepare e' la stessa di S
    SliceCDir = "/Users/cecilia/Documents/Documents_work/COVID/ZenodoNewPipe3/"+ CaseNumber + "C"
    CVti = "/Users/cecilia/Documents/Documents_work/COVID/ZenodoNewPipe3/"+ CaseNumber +'_C.vti' # COVID mask and lung
    SliceLMaskDir = "/Users/cecilia/Documents/Documents_work/COVID/ZenodoNewPipe3/"+ CaseNumber + "LM"
    # remove and create directories
    if os.path.exists(SliceCDir):
        sh.rmtree(SliceCDir)
    os.mkdir(SliceCDir)

    #CVcmd = "python3 main.py -m test -model /Users/cecilia/Documents/Documents_work/COVID/covseg-050321/covseg/newtrain-test1.ph -d1 {sliceldir} -d2 {slicecdir} -n {cvti} -i {originalvti} -f cpu -md {slicelmaskdir}".format(sliceldir = SliceLDir, slicecdir = SliceCDir, cvti = CVti, originalvti = OriginalVti, slicelmaskdir = SliceLMaskDir)
    cmd = "python3 main.py -m test -model segnet_norm_bestia_chinese_final.pth -d1 {sliceldir} -d2 {slicecdir} -n {cvti} -i {originalvti} -f cpu -md {slicelmaskdir}".format(sliceldir = SliceLDir, slicecdir = SliceCDir, cvti = CVti, originalvti = OriginalVti, slicelmaskdir = SliceLMaskDir)
    os.system(cmd)

