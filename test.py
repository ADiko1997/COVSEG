import torch
import model
import dataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import random
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import os 
from torch.autograd import Variable


np.random.seed(0)
random.seed(0)

torch.cuda.manual_seed(0)
torch.manual_seed(0)

def test_analyticaly(model, loader, batch_size, dir_, flag):

  model.eval()
  with torch.no_grad():
    for i, (images,  im) in enumerate(loader):
      #print(im)
      images = images.float()
      if flag == "gpu":
        images = images.cuda()
      output = model(images)
      # output_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy().astype(np.uint8) * 50
      output_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy().astype(np.uint8)
      #print("values: ",np.unique(output_mask, return_counts=True))

      # label_colors = np.array([(0,0,0),(128,0,0),(0,128,0), (0,0,128)])
      # r = np.zeros_like(output_mask).astype(np.uint8)
      # g = np.zeros_like(output_mask).astype(np.uint8)
      # b = np.zeros_like(output_mask).astype(np.uint8)

      # for l in range(0, 4):
      #   idx = output_mask == l
      #   r[idx] = label_colors[l, 0]
      #   g[idx] = label_colors[l, 1]
      #   b[idx] = label_colors[l, 2]

      # rgb = np.stack([r, g, b], axis=2)
      output_mask = np.asarray(output_mask)
      output_mask = np.flipud(output_mask)
      vtk_array = numpy_to_vtk(output_mask.ravel())
      i2d = vtkImageData()
      i2d.SetDimensions(512, 512, 1)
      i2d.SetSpacing(1,1,1)
      i2d.AllocateScalars(VTK_UNSIGNED_CHAR, 1)
      i2d.GetPointData().SetScalars(vtk_array)
      i2d.Modified()

      writer = vtkJPEGWriter()
      writer.SetInputData(i2d)
      writer.SetFileName( os.path.join(dir_, im[0]))
      writer.Write()
      print(os.path.join(dir_, im[0]))
    # print(" Val_Loss: %.4f" % (val_loss/i))


def load_weight_from_parallel(PATH:str, device=None):
  """
  input: PATH -> Path to trained model

  function: nn.Dataparallel models saves the dict_states as modules, meanwhile normal models saves them without modules
            so we want to remove the "module." part from the keys of the dictionary

  output: new dictionary with renamed keys
  """
  if device == "cpu":
    parallel_weights = torch.load(PATH, map_location=torch.device("cpu"))
  else:
    parallel_weights = torch.load(PATH)
  p_weights = dict(parallel_weights)
  keys = [k for k in p_weights.keys()]
  new_weights ={}
  for key in keys:
    new_weights[key[7:]] = p_weights[key]

  return new_weights


def dice_loss(pred, target, loss=True):

       """
       Calculate accuracy (customized on our needs) and returns dice coefitient as a loss parameter( with a backward function )

       inputs:
              @pred: Network output
              @targer: Labels
        
       output:
              @dice : dice coefition with autograd 
              @accuracy: accuracy for our purpose
       """
       pred = torch.argmax(pred, dim=1)

       pred = torch.where(pred == 2,1,0)
       target = torch.where(target == 2,1,0)

       init_ = torch.zeros(1)
       dice  = Variable(init_, requires_grad=True)

       numerator = 2 * torch.sum(pred * target)
       denominator = torch.sum(pred.sum() + target.sum())
       dice = 1 - ((numerator + 1) / (denominator + 1))
        
       return dice

def accuracy(pred, target, batchsize):

       pred = torch.argmax(pred, dim=1)

       pred = torch.where(pred == 2,1,0)
       target = torch.where(target == 2,1,0)

       
       dice  = torch.zeros(batchsize)

       for i in range(batchsize):

        numerator = 2*torch.sum(pred[i] * target[i])
        denominator = torch.sum(target[i].sum() + pred[i].sum())
        dice[i] = 1*(((numerator + 1) / (denominator + 1)) >= 0.8)

       
       return dice.sum()/batchsize