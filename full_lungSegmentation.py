import numpy as np 
import vtk 
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import SimpleITK as sitk
from lungmask import mask 
import argparse
from vtk import *

def segment(image, image2):
    #vti image attributes
    size = image.GetDimensions()
    spacing = image.GetSpacing()
    scalars = image.GetPointData().GetScalars()
    origin = image.GetOrigin()

    #extractin image as a numpy array
    scalars = vtk_to_numpy(scalars)
    scalars = scalars.reshape(size[2],size[1],size[0])

    #itk image from vti image 
    sitk_image = sitk.GetImageFromArray(scalars, sitk.sitkInt8)
    sitk_image.SetSpacing((spacing[0],spacing[1],spacing[2]))
    sitk_image.SetOrigin([origin[0],origin[1],origin[2]])

    #segment
    #model = mask.get_model('unet','R231CovidWeb')
    seg = mask.apply(sitk_image)
    seg = np.asarray(seg)
    unique = np.unique(seg)
    print(unique)

    #smoothed image
    #vti image attributes
    size2 = image2.GetDimensions()
    spacing2 = image2.GetSpacing()
    scalars2 = image2.GetPointData().GetScalars()
    origin2 = image2.GetOrigin()

    #extractin image as a numpy array
    scalars2 = vtk_to_numpy(scalars2)
    scalars2 = scalars2.reshape(size2[2],size2[1],size2[0])

    #itk image from vti image 
    sitk_image2 = sitk.GetImageFromArray(scalars2, sitk.sitkInt8)
    sitk_image2.SetSpacing((spacing2[0],spacing2[1],spacing2[2]))
    sitk_image2.SetOrigin([origin2[0],origin2[1],origin2[2]])


    #Find Seed
    coordinates = np.where(seg == 1)
    z_min, y_min, x_min = np.min(coordinates[0]), np.min(coordinates[1]), np.min(coordinates[2])
    z_max, y_max, x_max = np.max(coordinates[0]), np.max(coordinates[1]), np.max(coordinates[2])

    bbox2 = np.where(seg == 2)
    z_min2, y_min2, x_min2 = np.min(bbox2[0]), np.min(bbox2[1]), np.min(bbox2[2])
    z_max2, y_max2, x_max2 = np.max(bbox2[0]), np.max(bbox2[1]), np.max(bbox2[2])

    x_upper = x_max2 
    y_upper = max(y_max, y_max2)
    z_lower = min(z_min, z_min2)
    z_upper = max(z_max, z_max2)


    seeds = [0,0,0]
    for k in range(250, int(x_upper)):
        for j in range(250, y_upper):
            for i in range(z_lower, z_upper):
                if scalars2[i,j,k] == -1024:
                    if seeds[0] == 0:
                        seeds = [k,j,i]
    print(seeds)
    #Region Growing
    seg_explicit_thresholds = sitk.ConnectedThreshold(sitk_image2, seedList=[seeds], lower=-1025, upper=-940)


    #Create final mask
    trachea = sitk.GetArrayFromImage(seg_explicit_thresholds)
    trachea = np.asarray(trachea)
    print(np.unique(trachea, return_counts=True))
    trachea = np.where(trachea>0,2,0)
    seg = np.where(seg>0,1,0)
    #full_mask = np.add(seg, trachea)
    #full_mask = np.where(full_mask > 0,1,0)
    #full_mask = np.where(full_mask>2,2,full_mask)

    #Get Image data
    dimensions = sitk_image.GetSize() 
    spacing = sitk_image.GetSpacing()
    # data = sitk.GetArrayFromImage(sitk_image)
    # data = np.asarray(data)
    #new_image = np.multiply(data, full_mask)
    # new_image = full_mask

    #Create VTI image
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(dimensions)
    image.GetPointData().SetScalars(numpy_to_vtk(seg.ravel()))

    image2 = vtk.vtkImageData()
    image2.SetSpacing(spacing)
    image2.SetDimensions(dimensions)
    image2.GetPointData().SetScalars(numpy_to_vtk(trachea.ravel()))

    return image, image2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p','--path',help='vti image path')
    parser.add_argument('-t', '--trachea', help="where to save trachea mask")
    parser.add_argument('-l', '--lung', help="where to save lung mask")

    args = parser.parse_args()

    #Read
    path = args.path
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path) #inserite il vostro path
    reader.Update()
    image = reader.GetOutput()


    kernelsize = 2
    flt = vtkImageMedian3D()
    flt.SetKernelSize(kernelsize, kernelsize, kernelsize)
    # flt.SetInputData(reader.GetOutput())
    flt.SetInputData(image)
    flt.Update()

    #Segment
    image2 = flt.GetOutput()
    image , image2 = segment(image, image2)

    #Save
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(args.lung)
    writer.SetInputData(image)
    writer.Write()

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(args.trachea)
    writer.SetInputData(image2)
    writer.Write()
