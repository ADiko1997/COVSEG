#Use lungmask to extrac lung mask
import numpy as np 
import vtk 
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import SimpleITK as sitk
from lungmask import mask 
import argparse

def segment(image):
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
    print("segmenting")
    seg = mask.apply(sitk_image)
    seg = np.asarray(seg)
    unique = np.unique(seg)

    #Create final mask
    full_mask = np.where(seg > 0,1,0)
    background = np.where(full_mask==0,-2000,0)

    masks = vtk.vtkImageData()
    masks.SetSpacing(spacing)
    masks.SetDimensions(size)
    masks.GetPointData().SetScalars(numpy_to_vtk(full_mask.ravel()))

    data = sitk.GetArrayFromImage(sitk_image)
    data = np.asarray(data)
    new_image = np.multiply(data, full_mask)
    new_image = new_image + background

    #Create VTI image
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(size)
    image.GetPointData().SetScalars(numpy_to_vtk(new_image.ravel()))

    return masks, image



if __name__ == "__main__":

    print("enter")
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str,
                        help='vti_path')

    parser.add_argument('-m', '--mask', type=str,
                        help='mask path')

    parser.add_argument('-e', '--ext', type=str,
                        help='extracted lung')

    args = parser.parse_args()
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(args.path) #inserite il vostro path
    reader.Update()
    image = reader.GetOutput()

    mask, ext_lung = segment(image)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(args.mask)
    writer.SetInputData(mask)
    writer.Write()

    writer2 = vtk.vtkXMLImageDataWriter()
    writer2.SetFileName(args.ext)
    writer2.SetInputData(ext_lung)
    writer2.Write()
