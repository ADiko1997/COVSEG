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
    #model = mask.get_model('unet','R231CovidWeb')
    seg = mask.apply(sitk_image)
    seg = np.asarray(seg)
    unique = np.unique(seg)
    #print(unique)


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
                if scalars[i,j,k] == -1024:
                    if seeds[0] == 0:
                        seeds = [k,j,i]
    #print(seeds)
    #Region Growing
    seg_explicit_thresholds = sitk.ConnectedThreshold(sitk_image, seedList=[seeds], lower=-1025, upper=-990)


    #Create final mask
    trachea = sitk.GetArrayFromImage(seg_explicit_thresholds)
    trachea = np.asarray(trachea)
    full_mask = trachea

    #Get Image data
    dimensions = sitk_image.GetSize() 
    spacing = sitk_image.GetSpacing()
    data = sitk.GetArrayFromImage(sitk_image)
    data = np.asarray(data)
    new_image = np.multiply(data, full_mask)

    #Create VTI image
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(dimensions)
    image.GetPointData().SetScalars(numpy_to_vtk(new_image.ravel()))

    return image



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-p','--path',help='vti image path')
    parser.add_argument('-t', '--trachea', help="where to save trachea mask")

    args = parser.parse_args()

    #Read
    path = args.path
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path) #inserite il vostro path
    reader.Update()
    image = reader.GetOutput()

    #Segment
    image = segment(image)

    #Save
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(args.trachea)
    writer.SetInputData(image)
    writer.Write()
