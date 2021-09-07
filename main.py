import train_valid
import test
import argparse
import dataset
import torch 
from torch import nn 
import model
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import assemble_to_vti
import shutil as sh

print(torch.__version__)
np.random.seed(0)
random.seed(0)

torch.cuda.manual_seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('-p','--path',help='working directory')

parser.add_argument('-m', '--mode', help='Running modality')

parser.add_argument('-model', '--model',help='models name')

parser.add_argument('-d1', '--test_dir' ,help='image_slices')

parser.add_argument('-d2', '--save_dir' ,help='covid_predicted_slices')

parser.add_argument('-i', '--vti_path' ,help='original_vti')

parser.add_argument('-n', '--name_assamble' ,help='name of assambled vti covid masks')

parser.add_argument('-f', '--flag' ,help='running device')

parser.add_argument('-md', '--bkg_masks_dir', help="backgroung masks directory")



args = parser.parse_args()
PATH__ = args.path
mode = args.mode
model_weights = args.model
if mode == 'test':
    print(mode)
if mode == 'test' and  model_weights == None:
    print("Error: Model weights are missing")
    exit(0)
elif mode == 'test' and model_weights != None:
    # model_path = os.path.join("/home", model_weights)
    model_path = model_weights

class Config:
    n_epochs: int = 30  # number of epochs of training
    decay_epoch: int = 5  # epoch from which to start lr decay
 
    img_height: int = 512  # size of image height # default 256x256
    img_width: int = 512  # size of image width
 
    batch_size: int = 4 # size of the batches
    lr: float = 0.001  # adam: learning rate
    b1: float = 0.5  # adam: decay of first order momentum of gradient
 
    channels: int = 3  # number of image channels
    moemntum: int = 0.9
    weight_decay = 0.001
    lr_decay = 0.5


cfg = Config()

net = model.DeepLabV3()

if torch.cuda.is_available() and mode=='train':
    net = nn.DataParallel(net)
    net.cuda()
    weights = dataset.WeightInit(PATH__)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.cuda.FloatTensor(weights))
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.moemntum,
                                weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

elif torch.cuda.is_available() and mode !='train':
  model =  net.cuda()

if args.flag == "cpu":
    model = net.cpu()

if mode == "train":

    train_img = os.path.join(PATH__,"Zenodo_train")
    test_img = os.path.join(PATH__,"cov_data/train")

    trainsetCT__ = dataset.COVIDDataSet(root=train_img,
                       img_transform= True, mode='train')
 
    valsetCT__ = dataset.COVIDDataSet(root=test_img,
                     img_transform= True)

    train_loader = DataLoader(trainsetCT__, batch_size=cfg.batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(valsetCT__, batch_size=cfg.batch_size, drop_last=True, shuffle=True)
    train_valid.train(net, cfg, val_loader, train_loader, optimizer, criterion, scheduler)

else:
    test_img = args.test_dir
    save_dir = args.save_dir
    bkg_masks_dir =  args.bkg_masks_dir
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    else:
        sh.rmtree(save_dir)
        os.mkdir(save_dir)

    testSetCT__ = dataset.COVIDDataSetTest(root=test_img, bkg_mask_dir= bkg_masks_dir, 
                     img_transform= True)
    
    weights =test.load_weight_from_parallel(model_path, args.flag)
    model.load_state_dict(weights)
    # model.load_state_dict(torch.load(model_path, args.flag))
    val_loader = DataLoader(testSetCT__, batch_size=1, drop_last=False, shuffle=False) 
    test.test_analyticaly(model, val_loader, cfg.batch_size, save_dir, args.flag) 

    cmd = "python3 assemble_to_vti.py -d {save_dir} -n {name_assamble} -v {vti_path}".format(save_dir=args.save_dir, 
                                                                                    name_assamble=args.name_assamble, 
                                                                                    vti_path=args.vti_path
                                                                                    )
    os.system(cmd)

# Example of use
# python3 main.py -m test -model /home/sm/COV19/segnet_norm.pth -d1 /home/sm/COV19/SEGMENTED_SLICES171.1 -d2 /home/sm/COV19/CASE171.1_PRED/ -n covid_mask_case171.1.vti -i /home/sm/COV19/CASE171.1.vti
