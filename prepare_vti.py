import argparse
import os 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--path', type=str, help='vti_path') #original case VTI
    parser.add_argument('-m', '--mask', type=str, help='mask path') #where vti file of mask will be saved
    parser.add_argument('-d1', '--dir_for_image', type=str, help='image slices dir name')
    parser.add_argument('-d2', '--dir_for_mask', type=str, help='mask slices dir name')
    parser.add_argument('-n', '--new', type=str, help='dir where new images will reside')
    parser.add_argument('-e', '--ext', type=str, help='dir where extracted lung will reside')
    parser.add_argument('-z', '--zenodo', type=str, help='if case is zenodo')



    args = parser.parse_args()
    ZENODO = args.zenodo
    #segment 

    if not ZENODO==1:

        cmd = "python3 extract_mask.py -p {vti_path} -m {mask_vti} -e {extracted_lung_vti}".format(vti_path=args.path, mask_vti=args.mask, extracted_lung_vti=args.ext) 
        os.system(cmd)

    #extract lung

    #slice
    cmd1 = "python3 slice_vti.py -p {vti_path} -d {dir_path} -m 0 -s {mask_path}".format(vti_path=args.ext, dir_path=args.dir_for_image, mask_path=args.mask) #vti segmetato
    cmd2 = "python3 slice_vti.py -p {vti_path} -d {dir_path} -m 1".format(vti_path=args.mask, dir_path=args.dir_for_mask) #mask path

    os.system(cmd1)
    os.system(cmd2)

    #normalize and transform to 8 bit
    #cmd = "python3 grayscale_adj.py -p {img_slices} -m {mask_slices} -n {segmented_image_slices}".format(img_slices = args.dir_for_image, mask_slices=args.dir_for_mask, segmented_image_slices=args.new)
    #os.system(cmd)

# example how to run it
# python3 prepare_vti.py -i /home/sm/COV19/CASE59.vti -m /home/sm/COV19/CASE59_mask.vti -d1 /home/sm/COV19/CASE59 -d2 /home/sm/COV19/CASE59_MASK -n /home/sm/COV19/SEGMENTED_SLICES
