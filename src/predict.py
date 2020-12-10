from helpers import *
import pickle 
import torch
import torchvision.models as models
from torch.autograd import Variable
# import onnxruntime
import argparse
import torch
# import joblib
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from glob import glob
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset
#from torchsummary import summary
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import numpy as np
import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os
import itertools
from pathlib import Path

# Image manipulations
from PIL import Image

from timeit import default_timer as timer
import time
import datetime
from datetime import datetime

# Visualizations
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['font.size'] = 14

from torch import Tensor, nn
from torch.nn.functional import interpolate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', default='', help='job name')
    parser.add_argument('--model_fn', default='', help='best model checkpoint path')
    parser.add_argument('--test_dir', default='', help='image foler. Folder contains folders of images with folder name as class name')
    parser.add_argument('--result_base_dir', default='', help='Result directory. A subdir named after your jobname will be created.')
    parser.add_argument('--img_size', type=int, default=224)
    opt = parser.parse_args()
    
    #https://pytorch.org/docs/stable/torchvision/models.html
#     model_choices=['googlenet','resnet50','squeezenet', 'vgg16', 'wide_resnet50_2']
    
    #### Initialize variables
    jobname=opt.jobname
    model_fn = opt.model_fn
    test_dir = opt.test_dir
    result_base_dir=opt.result_base_dir
    img_size=opt.img_size


    ################ create result directory
    result_dir=result_base_dir+"/"+jobname
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # If there's a GPU available...
    test_on_gpu = cuda.is_available()
    if test_on_gpu:    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        # Number of gpus
        gpu_count = cuda.device_count()
        print('There are %d GPU(s) available.' % gpu_count)

        print('We will use the GPU:', torch.cuda.get_device_name(0))

        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False    

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        multi_gpu = False  
    print(test_on_gpu,multi_gpu)

    # Load Checkpoint
    def load_checkpoint(path):
        """Load a PyTorch model checkpoint

        Params
        --------
            path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """
        # Load in checkpoint
        checkpoint = torch.load(path)
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

        # Load in the state dict
        model.load_state_dict(checkpoint['state_dict'])

        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} total gradient parameters.')

        if test_on_gpu:
            model = model.to('cuda')

        else:
            model = model

        # Model basics
        model.class_to_idx = checkpoint['class_to_idx']
        model.idx_to_class = checkpoint['idx_to_class']
        model.epochs = checkpoint['epochs']

        # Optimizer
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    checkpoint_path = model_fn
    model, optimizer = load_checkpoint(path=checkpoint_path)
    class_names = list(model.class_to_idx.keys())
    # print('class_names',class_names)


    # Define Paths and params
    # fullDatasetdir = test_dir


    # Image transformations
    # https://towardsdatascience.com/improves-cnn-performance-by-applying-data-transformation-bf86b3f4cef4
    ##########################
    image_transforms = {
        'test':
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Define DataLoaders
    print('test_dir',test_dir)
    try:
        print('with paths')
        test_dataset = ImageFolderWithPaths(root=test_dir, transform=image_transforms['test']) # our custom dataset with filenames

    except RuntimeError:
        print('run custom dataset')
        test_dataset = CustomDataSet(test_dir, transform=image_transforms['test']) # our custom dataset with filenames
        # continue 
    # Dataloader iterators, make sure not to shuffle
    if test_on_gpu == True:

        # inf_data = InfDataloader(img_folder=data['blind_test'], target_size=img_size)
        # inf_dataloader = DataLoader(inf_data, shuffle=False, num_workers=10)

        dataloaders = {
            'blind_test': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,num_workers=10),
        }

    else:
        # inf_data = InfDataloader(img_folder=data['blind_test'], target_size=img_size)
        # inf_dataloader = DataLoader(inf_data, shuffle=False, num_workers=0)
        dataloaders = {
                'blind_test': DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,num_workers=0),
            }  

    # with open('./src/output/test_dataloader.pkl', 'wb') as to_write: #dummied clustering features
    #     pickle.dump(dataloaders['blind_test'], to_write)

    idx_to_class = model.idx_to_class
    # print('idx_to_class',idx_to_class)
    


    ##### save all blind test image prediction result
    blind_test_truelabels = []
    blind_test_predictions = []
    blind_test_probs = []
    ImageFile = []
        
    model.eval()
    with torch.no_grad():
        print("Getting predictions from blind test set...")
        try:
            for data, targets, paths in dataloaders['blind_test']:
                # Tensors to gpu
                if test_on_gpu:
                    data, targets = data.to('cuda'), targets.to('cuda')           
                for label in targets.cpu().numpy():
    #                 print(label)
                    blind_test_truelabels.extend([idx_to_class[label]])
                test_output = model(data)
                # print('test_output',test_output)
                test_pred_prob = torch.exp(test_output)
                # test_pred = torch.max(test_pred_prob, dim=1)
                _, test_pred = torch.max(test_output, dim=1)
                blind_test_predictions.extend([idx_to_class[val] for val in test_pred.cpu().numpy()])
                blind_test_probs.extend(test_pred_prob.cpu().numpy())
                ImageFile.extend(paths)
        except:
            for data, paths in dataloaders['blind_test']:
                # Tensors to gpu
                if test_on_gpu:
                    data = data.to('cuda')       
                test_output = model(data)
                # print('test_output',test_output)
                test_pred_prob = torch.exp(test_output)
                # test_pred = torch.max(test_pred_prob, dim=1)
                _, test_pred = torch.max(test_output, dim=1)
                blind_test_predictions.extend([idx_to_class[val] for val in test_pred.cpu().numpy()])
                blind_test_probs.extend(test_pred_prob.cpu().numpy())
                ImageFile.extend(paths)
            # continue

    try:
        blind_test_pred_df = pd.DataFrame({'pred_prob': blind_test_probs,'pred':blind_test_predictions,'actual':blind_test_truelabels, 'img_file':ImageFile} ) 
        blind_test_pred_df.to_csv(f'{result_dir}/blind_test_pred.csv',index=None)   
    except:
        blind_test_pred_df = pd.DataFrame({'pred_prob': blind_test_probs,'pred':blind_test_predictions, 'img_file':ImageFile} ) 
        blind_test_pred_df.to_csv(f'{result_dir}/blind_test_pred.csv',index=None)   
        # continue
    # blind_test_pred_df = pd.DataFrame({'pred':blind_test_predictions,'actual':blind_test_truelabels, 'img_file':ImageFile} ) 
    print(len(blind_test_pred_df))
    print(f"Prediction output of all images saved to {result_dir}/blind_test_pred.csv")
    