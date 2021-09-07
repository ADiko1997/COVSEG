from vtk import *
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import os
import argparse
import shutil as sh

def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    Parameters
    ----------
    img: numpy.ndarray[np.uint16]
        image that should be mapped
    lower_bound: int, optional
        lower bound of the range that should be mapped to ``[0, 255]``,
        value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
        (defaults to ``numpy.min(img)``)
    upper_bound: int, optional
       upper bound of the range that should be mapped to ``[0, 255]``,
       value must be in the range ``[0, 65535]`` and larger than `lower_bound`
       (defaults to ``numpy.max(img)``)

    Returns
    -------
    numpy.ndarray[uint8]
    '''
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        help='vti_path')

    parser.add_argument('-d', '--dir', type=str,
                        help='dir name')

    parser.add_argument('-m', '--mask', type=int,
                        help='is_mask')

    parser.add_argument('-s', '--seg', type=str,
                        help='mask')

    args = parser.parse_args()

    reader = vtkXMLImageDataReader()
    reader.SetFileName(args.path)
    reader.Update()

    # a is the original image array of 16 bit
    a = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
    dim = reader.GetOutput().GetDimensions()

    a = a.reshape(dim)

   

    #extract values we want to delete
    # bigger_values = np.where(a>200,0,1)
    # smaller_values = np.where(a<-1050,0,1)
    # values_to_remove = np.multiply(bigger_values, smaller_values)


    # newmax=200-a.min()
    # newmin = a - (-1050)
    # a = np.where(a > 0, 0, a)
    # a = a - a.min()
    #reverse values
    
    
    if  args.mask == 0:
        # print("enters")
        #  a = -a 
         reader = vtkXMLImageDataReader()
         reader.SetFileName(args.seg)
         reader.Update()

        # a is the original image array of 16 bit
         mask = vtk_to_numpy(reader.GetOutput().GetPointData().GetScalars())
         dims = reader.GetOutput().GetDimensions()
         mask = mask.reshape(dims)
         #for x in np.nditer(a):
            #print(0," ",x)
         a = np.where(a >200, -1050, a)
         a = np.where(a<-1050,-1050, a)
         a = a+1050
         #for y in np.nditer(a):
             #print(1," ",y)
         amax = 1250
         amin = 0
#        a8 = map_uint16_to_uint8(a, lower_bound=a.min(), upper_bound=a.max())
         a8 = map_uint16_to_uint8(a, lower_bound=amin, upper_bound=amax) #CV
         #for z in np.nditer(a8):
              #print(2," ",z)
        #  a8 = np.multiply(a8, values_to_remove).astype(np.uint8)
         a8 = np.multiply(a8, mask).astype(np.uint8)
         #for k in np.nditer(a8):
              #print(3," ",k)
    
    else:
        a8 = a.astype(np.uint8)
        print('mask')
    new_dir = os.path.join(os.getcwd(),args.dir)
    if os.path.exists(new_dir):
        sh.rmtree(new_dir)
    os.mkdir(new_dir)


    os.chdir(new_dir)

    for k in range(dim[2]):

        b8 = a8.reshape(dim[2], dim[1], dim[0])
        linear_array = b8[k,:,:].ravel()

        if k<10:
            print(np.unique(linear_array, return_counts=True))
        vtk_array = numpy_to_vtk(linear_array)

        i2d = vtkImageData()
        i2d.SetDimensions(dim[0], dim[1], 1)
        i2d.SetSpacing(reader.GetOutput().GetSpacing())
        i2d.AllocateScalars(VTK_UNSIGNED_CHAR, 1)
        i2d.GetPointData().SetScalars(vtk_array)
        i2d.Modified()

        writer = vtkJPEGWriter()
        writer.SetInputData(i2d)
        writer.SetFileName( '%d.jpeg' % k)
        writer.Write()

