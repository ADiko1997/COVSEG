from vtk import *
import numpy as np 
import os 
import argparse
import cv2
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def assemble(dir_images, name_vti, vti_image):

    reader = vtkXMLImageDataReader()
    reader.SetFileName(vti_image)
    reader.Update()
    spacing = reader.GetOutput().GetSpacing()
    dim = reader.GetOutput().GetDimensions()

    images = os.listdir(dir_images)
    # dir_images = "/home/manash/COV19/CASE171.1"
    indexes = []
    for i in images:
        indexes.append(int(i.replace('.jpeg',"")))

    indexes, images = zip(*sorted(zip(indexes, images)))
    # print(len(images))
    full_mask = np.ones((1,512,512)).astype(np.uint8)

    for i in images:
        image = cv2.imread(os.path.join(dir_images, i),cv2.IMREAD_GRAYSCALE)
        image = np.asarray(image).astype(np.uint8)
        image = image.reshape(1, 512, 512)
        full_mask = np.append(full_mask, image, axis=0)

    full_mask = full_mask[1:,:,:].copy()
    # full_mask = np.where(full_mask>1,1,0)
    # print(full_mask.shape)
    full_mask = full_mask.reshape(dim[2], dim[1], dim[0])
    # full_mask = np.swapaxes(full_mask, 0,2)
    vtk_full_mask = numpy_to_vtk(full_mask.ravel())
    image = vtkImageData()
    image.SetDimensions(dim[0],dim[1],dim[2])
    image.SetSpacing(spacing[0], spacing[1], spacing[2])
    image.GetPointData().SetScalars(vtk_full_mask)

    writer = vtkXMLImageDataWriter()
    #print(os.path.join(dir_images, name_vti)
    # writer.SetFileName(os.path.join(dir_images, name_vti))
    writer.SetFileName(name_vti)

    writer.SetInputData(image)
    writer.Write()

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir_images', type=str,help='path to jpeg folder')
    parser.add_argument('-n', '--name_vti', type=str, help='Name of new vti mask')
    parser.add_argument('-v', '--vti', type=str, help='original vti path')

    args = parser.parse_args()

    assemble(args.dir_images, args.name_vti, args.vti)