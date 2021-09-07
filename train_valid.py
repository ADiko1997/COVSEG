import torch
import model
import dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from torch.autograd import Variable
from test import *
from vtk import *
np.random.seed(0)
random.seed(0)

torch.cuda.manual_seed(0)
torch.manual_seed(0)

# +
def train(model, cfg, val_loader, train_loader, optimizer, criterion, scheduler):

    if True:
        model.train()
        for epoch in range(cfg.n_epochs):
            running_loss = 0.0
            DICE = 0.0
            tr_accuracy = 0.0
            for i, (images, labels_group,im, masks ) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = images.float()
                    images = images.cuda()
                    labels_group = labels_group.long()
                    labels_group = labels_group.cuda()
                else:
                    continue
            
                optimizer.zero_grad()
                output = model(images)
                labels_group = labels_group.view(cfg.batch_size,512,512)

                dice =dice_loss(output, labels_group)
                loss = criterion(output.cuda(), labels_group.cuda())
                loss = loss + dice/cfg.batch_size
                DICE+=1 - dice/cfg.batch_size
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                tr_accuracy += accuracy(output.cuda(), labels_group.cuda(), cfg.batch_size)
    #             print(1 - dice/cfg.batch_size)
            
                    # lr = lr * (1-(92*epoch+i)/max_iters)**0.9
                    # for parameters in optimizer.param_groups:
                    #     parameters['lr'] = lr
            
            print("Epoch [%d] Loss: %.4f  DICE: %.4f Accuracy: %.4f " % (epoch+10, running_loss/len(train_loader) , DICE/len(train_loader), tr_accuracy/len(train_loader)))
            
            if (epoch+1) %2 ==0:
                with torch.no_grad():
                    val_loss = 0.0
                    val_DICE = 0.0
                    val_accuracy = 0.0
                    for i, (images, labels_group, im, masks) in enumerate(val_loader):
                        images = images.float()
                        images = images.cuda()
                        labels_group = labels_group.long()
                        labels_group = labels_group.cuda()

                        output = model(images)
                        labels_group = labels_group.view(cfg.batch_size,512,512)
                        dice= dice_loss(output, labels_group)
                        loss = criterion(output.cuda(), labels_group.cuda())
                        val_loss+=loss+dice/cfg.batch_size
                        val_DICE += 1 - dice/cfg.batch_size

                        val_accuracy += accuracy(output.cuda(), labels_group.cuda(), cfg.batch_size)
                        
                    print("Epoch [%d] Val_Loss: %.4f DICE: %.4f Accuracy: %.4f" % (epoch+10, val_loss/len(val_loader), val_DICE/len(val_loader), val_accuracy/len(val_loader)))
                    torch.save(model.state_dict(), "/home/dikoanxh/COVSEG/segnet_norm_bestia_chinese_final.pth")




