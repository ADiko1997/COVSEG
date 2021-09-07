import os
import numpy as np
import argparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import math
from vtk import *
import shutil as sh


def calc_dev_nean(img_path, mask_path):

    medie = []
    dev = []
    for i in os.listdir(mask_path):
        
        # print("path :",i)
        mask_filename = os.path.join(mask_path + i) 
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(mask_filename)
        reader.Update()
        mask = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
        mask = mask.flatten()

        image_filename = os.path.join(img_path + i)
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(image_filename)
        reader.Update()
        im_Array = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
        im_Array = im_Array.flatten()

        # im_Array= 255*((im_Array-im_Array.min())/(im_Array.max()-im_Array.min())) + 10 

        
        im_Array=im_Array*(mask>0)*1
                ##a = im_Array[im_Array>0].ravel() #Like flatten
        # im_Array=im_Array*mask
        # adjdev= (im_Array-102.1807)*5.5

                ##im_Array = ((im_Array-np.mean(a))*(44/np.std(a)))+99
        # im_Array=im_Array+adjdev


        pix0=np.sum((mask==0)*1)


        arrcin=im_Array.flatten().tolist()
        arrcin.sort()
        newarr=arrcin[pix0:]
        newarr=np.array(newarr)
        newarr=newarr.astype(np.uint8)
        newarr = newarr[newarr >0].ravel()
        mean = np.mean(newarr)
        std = np.std(newarr)
        medie.append(mean)

        if math.isnan(std):
            dev.append(0)
        else:
            dev.append(std)

    return medie, dev
        

def gaussian(img_path, mask_path ,new_img_path, dev, medie):
    k = 0

    if not os.path.exists(new_img_path):
        os.mkdir(new_img_path)
    
    else:
        sh.rmtree(new_img_path)
        os.mkdir(new_img_path)


    for i in os.listdir(mask_path):
        # print(i)
        mask_filename = os.path.join(mask_path + i) 
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(mask_filename)
        reader.Update()
        mask = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
        dim = reader.GetOutput().GetDimensions()
        mask = mask.flatten()

        image_filename = os.path.join(img_path+ i)
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(image_filename)
        reader.Update()
        im_Array = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
        im_Array = im_Array.flatten()
        
        # im_Array=(255*((im_Array-im_Array.min())/(im_Array.max()-im_Array.min())))
       
##CV        if  dev[k]!=0:
##CV          im_Array = 99 + (im_Array - medie[k]) * (44/dev[k])
        
        # print(np.mean(im_Array),np.std(im_Array))
        im_Array=im_Array*(mask>0)
        # im_Array=im_Array*mask
        im_Array = im_Array.astype(np.uint8)
        
        i2d = vtk.vtkImageData()
        i2d.SetDimensions(dim[0], dim[1], 1)
        i2d.SetSpacing(reader.GetOutput().GetSpacing())
        i2d.AllocateScalars(VTK_UNSIGNED_CHAR, 1)
        i2d.GetPointData().SetScalars(numpy_to_vtk(im_Array.ravel()))
        i2d.Modified()

        writer = vtk.vtkJPEGWriter()
        writer.SetInputData(i2d)
        new_path = os.path.join(new_img_path,i)
        writer.SetFileName( new_path)
        writer.Write()

        k = k + 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='path of original slices')

    parser.add_argument('-m', '--mask', type=str,
                        help='dir where mask slices reside')

    parser.add_argument('-n', '--new', type=str,
                        help='dir where new images will reside')

    args = parser.parse_args()

    img_path = args.path
    mask_path = args.mask
    new_img_path = args.new

    if mask_path[-1] != "/":
        mask_path = mask_path+"/"

    if img_path[-1] != "/":
        img_path = img_path+"/"

    if new_img_path[-1] != "/":
        new_img_path = new_img_path+"/"

    mean, std = calc_dev_nean(img_path, mask_path)
    # print(std) 
    gaussian(img_path, mask_path, new_img_path, std, mean)
