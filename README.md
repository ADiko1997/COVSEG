# COVID-19 SEGMENTATION

## USE FOR EXTRACTIN COVID MASKS FROM VTI CT_IMAGES

#### 1- It is necessary to extract the lung and slice the image , using the pipeline can be done the following way:
Let`s say that we want to slice case 174 and the vti file is named CASE174.vti
The file that will do the required task is named prepare_vti.py and the command to call it is the following:

`python3 prepare_vti.py -i {path_to/CASE174.vti} -m {path_to/CASE174_mask.vti} -d1 {path_to/CASE174} -d2 {path_to/CASE59_MASK} -n {path_to/SEGMENTED_SLICES} -e {path_to/segmented_lung174.vti}`

- -i path to vti image 
- -m path to and the name you want to save the 3d mask 
- -d1 dir where you want to save original image slices 
- -d2 dir where you want to save mask slices  
- -n dir where you want to save segmented slices 
- -e dir where you want to save segmented lung before slicing it 


#### 2- Run the COV-19 Segmentation network with the following command

`python3 main.py -m {test} -model {path_to/segnet_norm.pth} -d1 {path_to/SEGMENTED_SLICES} -d2 {path_to/CASE174_PRED/} -n {covid_mask_case174.vti} -i {path_to/CASE174.vti} -f {device} -md {path_to/Sliced_Mask}`

- -m running mode, in this case should always be test, it changes only if u want to train
- -model path to the model weights
- -d1 path to segmented slices obtained from step 1
- -d2 path to dir where you want to save the network prediction and the assambled vti file 
- -n name of the assambled vti 
- -i path to the original vti
- -f runnning device,  cpu or gpu

#### Extract Trachea

`python3 trachea_segmentation.py -p {path to original vti} -t {path where to save the trachea segmented vti}`

#### Extract Trachea

Extract Lung and Trachea on the same vti

`python3 full_lungSegmentation.py -p {path to original vti} -t {path where to save the trachea segmented vti} -l {path where to save lung mask}`

