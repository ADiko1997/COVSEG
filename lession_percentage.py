from vtk import *
import numpy as np
import argparse
from vtk.util.numpy_support import vtk_to_numpy

def read_mask(filename):
    if ".nii" in filename:
        reader = vtkNIFTIImageReader()
    
    else:
        reader = vtkXMLImageDataReader()

    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput().GetPointData().GetScalars()
    data = vtk_to_numpy(data)

    return data

def calc_precentage(data, filename):
    #calc tot pixels
    print("Data",np.unique(data, return_counts=True))
    tot = np.where(data>0,1,0)
    print(np.unique(tot, return_counts=True))
    tot = np.unique(tot, return_counts=True)[1][1]
    print("TOT: ",tot)

    #calc lession pixels
    if ".nii" in filename:
        lession = np.where(data>2,1,0)
    else:
        lession = np.where(data>1,1,0)
    print(np.unique(lession, return_counts=True))
    lession = np.unique(lession, return_counts=True)[1][1]
    print("Lession: ",lession)

    #calc percentage
    perc = (lession/tot) *100

    #define interval
    if perc == 0:
        print("Helthy lung")

    elif perc >0 and perc <= 20:
        print("Lession percentage in 0-20%")

    elif perc > 20 and perc <= 40:
        print("Lession percentage in 21-40%")

    elif perc > 40 and perc <=60:
        print("Lession percentage in 41-60%")

    elif perc > 60 and perc <=80:
        print("Lession percentage in 61-80%")

    elif perc > 80 and perc <=100:
        print("Lession percentage in 81-100%")

    return perc
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, help='vti_path')
    args = parser.parse_args()

    #Get data 
    data = read_mask(args.path)

    #calc percentage
    perc = calc_precentage(data, args.path)
    print("Exact percentage:", perc)

