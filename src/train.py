#model training##
# https://github.com/MLWhiz/data_science_blogs/blob/master/compvisblog/Boats_Model.ipynb
################
import argparse
import torch
import pickle
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
from torch.autograd import Variable
# import onnxruntime
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
    parser.add_argument('--new_data_dir', default='', help='image folder. New data directory. A new folder structure with train, validation and test images')
    parser.add_argument('--result_base_dir', default='', help='Result directory. A subdir named after your jobname will be created.')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('-o','--max_epochs_stop', type=int, default=10, help = 'maximum number of epochs with no improvement in validation loss for early stopping')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--cnn_sel', type=str, default='resnet50', help='choice of cnn architecture vgg,resnet,densenet,inceptionnet,xceptionnet,alexnet,cnn1,cnn2,lenet5')
    parser.add_argument('--batch_size', type=int, default=16)
    # parser.add_argument('--class_type', type=str, default='multiclass', help='multiclass or multilabel')
    #parser.add_argument('--crop_dim', nargs='+', type=int, default=[0,0,0,0], help='crop pixels. example: --crop_dim 290 370 500 500. Image not cropped if not provided')
    #parser.add_argument('--transfer', action='store_true', help='perform transfer learning.')
    #parser.add_argument('--transfer_model', type=str, default='', help='transfer model filename')
    # parser.add_argument('--grayscale', action='store_true', help='train image in grayscale')
    #parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    # parser.add_argument('--target_train_size', type=int, default=10000, help='Target training size (augmentation purpose. Default 10000')
    opt = parser.parse_args()
    
    #https://pytorch.org/docs/stable/torchvision/models.html
    model_choices=['googlenet','resnet50','squeezenet', 'vgg16', 'wide_resnet50_2']
    
    #### Initialize variables
    jobname=opt.jobname
    new_data_dir=opt.new_data_dir
    result_base_dir=opt.result_base_dir
    epochs=opt.epochs
    max_epochs_stop = opt.max_epochs_stop
    img_size=opt.img_size
    cnn_sel=opt.cnn_sel
    batch_size=opt.batch_size
    # class_type=opt.class_type
    #crop_dim=opt.crop_dim
    #transfer=opt.transfer
    #transfer_model=opt.transfer_model
    # grayscale=opt.grayscale
    # target_train_size=opt.target_train_size


    ################ create result directory
    result_dir=result_base_dir+"/"+jobname
    Path(result_dir).mkdir(parents=True, exist_ok=True)


    save_file_name = f'{result_dir}/resnet50-transfer.pt'
    checkpoint_path = f'{result_dir}/resnet50-transfer.pth' 

    ####### Start logging
    
    flog = open(result_dir+"/pytorch_resnet.log","a")
    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    splitStr='\n\n\n---------------------Start Run pytorch_resnet-------------------------------------\n'
    flog.write(splitStr)
    flog.write(date_time)

    ##### Write log file
    datestr="date and time: "+date_time
    
    
    str0="Input parameters:"
    str0 = 'jobname | epochs | img_size | cnn_sel | batch_size' .\
                    format(jobname, new_data_dir, result_base_dir, epochs, img_size, cnn_sel,batch_size)
    # str0 = "%s\njobname: %s\original_data_dir: %s\new_data_dir: %s\nresult_base_dir: %s\epochs: %d\max_epochs_stop: %s\img_size: %d\n" % (str0,jobname,original_data_dir,new_data_dir,result_base_dir,epochs,max_epochs_stop,img_size)
    # str0="%s\n\cnn_sel: %d\nbatch_size:" % (str0,cnn_sel,batch_size)
    flog.write(str0)

    #### save input parameters
    d={'dts': date_time,'jobname':jobname, 'new_data_dir':new_data_dir, 'result_base_dir':result_base_dir,'epochs':epochs, 'max_epochs_stop':max_epochs_stop, 'img_size':img_size,'cnn_sel':cnn_sel,'batch_size':batch_size}
    param_df=pd.DataFrame(data=d,index=[0])
    param_df.to_csv(result_dir+"/param.csv",index=0)

    def imshow_tensor(image, ax=None, title=None):
        """Imshow for Tensor."""
        plt.figure(figsize=(8, 6))
        # if ax is None:
        #     fig, ax = plt.subplots()

        # Set the color channel as the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Reverse the preprocessing steps
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Clip the image pixel values
        image = np.clip(image, 0, 1)

        plt.imshow(image)
        # plt.savefig(f"{result_dir}/random_image.png")
        plt.imsave(f"{result_dir}/random_image.png",image)
        plt.axis('off')
        plt.close()

        return plt


    # Define Paths and params
    traindir = f"{new_data_dir}/train"
    validdir = f"{new_data_dir}/val"
    testdir = f"{new_data_dir}/test"

    # Whether to train on a gpu
    # If there's a GPU available...
    train_on_gpu = cuda.is_available()
    if train_on_gpu:    

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
    print(train_on_gpu,multi_gpu)

    # Image transformations
    # https://towardsdatascience.com/improves-cnn-performance-by-applying-data-transformation-bf86b3f4cef4
    ##########################
    image_transforms = {
        # Train uses data augmentation
        'train':
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
#             transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
#             transforms.Pad(25, padding_mode='symmetric'), # equals to zooming out the image. It will create a pad outside the given image on all sides with the certain value.
#             transforms.RandomRotation(degrees=15),
#             transforms.ColorJitter(), # Randomly change the brightness, contrast and saturation of an image.
#             transforms.RandomHorizontalFlip(),
#             transforms.CenterCrop(size=224),  # Image net standards; similar to zooming in the center of image. It will crop the given image to a desired output size and position
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], #All pre-trained models expect input images normalized in the same way (https://pytorch.org/docs/stable/torchvision/models.html)
                                [0.229, 0.224, 0.225]),  # Imagenet standards
#             transforms.RandomErasing(p=0.75,scale=(0.02, 0.1),value=1.0, inplace=False)        
        ]),
        # Validation does not use augmentation
        'valid':
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
            # Test does not use augmentation
        'test':
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

#     wt_train_ex_img = Image.open('/home/valerie_lim/Emerson/WireTagdata/train/YellowTag/YellowTag (98).jpg')
#     wt_test_ex_img = Image.open('/home/valerie_lim/Emerson/WireTagdata/test/CommissioningTag/CommissioningTag (95).jpg')

#     fnb_test_ex_img= Image.open('/home/valerie_lim/Emerson/FnBdata/test/316_CS_Bolt/316_CS_Bolt (33).jpg')
#     fnb_train_ex_img= Image.open('/home/valerie_lim/Emerson/FnBdata/train/316_CS_Bolt/316_CS_Bolt (1).jpg')

#     if original_data_dir.split('/')[-2] == 'FlangeAndBolt':
#         plt.figure(figsize=(24, 24))

#         for i in range(16):
#             ax = plt.subplot(4, 4, i + 1)
#             _ = imshow_tensor(image_transforms['test'](fnb_test_ex_img), ax=ax)

#         plt.savefig(f'{result_dir}/test_transformed_316_cs_33.png')
#         plt.close()

#         for i in range(16):
#             ax = plt.subplot(4, 4, i + 1)
#             _ = imshow_tensor(image_transforms['train'](fnb_train_ex_img), ax=ax)

#         plt.savefig(f'{result_dir}/train_transformed_316_cs_1.png')
#         plt.close()
        
#     elif original_data_dir.split('/')[-2] == 'WireTag':
#         plt.figure(figsize=(24, 24))

#         for i in range(16):
#             ax = plt.subplot(4, 4, i + 1)
#             _ = imshow_tensor(image_transforms['test'](wt_test_ex_img), ax=ax)

#         plt.savefig(f'{result_dir}/test_transformed_ct_95.png')
#         plt.close()

#         for i in range(16):
#             ax = plt.subplot(4, 4, i + 1)
#             _ = imshow_tensor(image_transforms['train'](wt_train_ex_img), ax=ax)

#         plt.savefig(f'{result_dir}/train_transformed_yt_98.png')
#         plt.close()        

#     else:
#         '''
#         current script is built to augment FnB and Wiretag only
#         '''
#         print('error. invalid folder name to perform augmentation')
    # Define DataLoaders

    data = {
        'train':
        datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
        datasets.ImageFolder(root=validdir, transform=image_transforms['valid']),
        'test':
        datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators, make sure to shuffle
    if train_on_gpu == True:
        # 
        dataloaders = {
            'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,num_workers=10), #shuffling the trainloader only, and not the val&testloader.
            'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=False,num_workers=10), 
            'test': DataLoader(data['test'], batch_size=len(data['test']), shuffle=False,num_workers=10),
        }

    else: 
        dataloaders = {
            'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True,num_workers=0), #shuffling the trainloader only, and not the val&testloader.
            'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=False,num_workers=0),
            'test': DataLoader(data['test'], batch_size=len(data['test']), shuffle=False,num_workers=0),
        }

    # Iterate through the dataloader once
    trainiter = iter(dataloaders['train'])
    features, labels = next(trainiter)
#     print(features.shape, labels.shape)

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)
        
    n_classes = len(categories)
    print(f'There are {n_classes} different classes.')


    # # Some Statistics
    class_to_idx = data['train'].class_to_idx
    idx_to_class = {
        idx: class_
        for class_, idx in data['train'].class_to_idx.items()
    }

    train_cnts = Counter([idx_to_class[x] for x in data['train'].targets])
    val_cnts = Counter([idx_to_class[x] for x in data['valid'].targets])
    test_cnts = Counter([idx_to_class[x] for x in data['test'].targets])

    train_cnts = pd.DataFrame({'cat' :list(train_cnts.keys()), 'train_cnt': list(train_cnts.values())})
    val_cnts = pd.DataFrame({'cat' :list(val_cnts.keys()), 'val_cnt': list(val_cnts.values())})
    test_cnts = pd.DataFrame({'cat' :list(test_cnts.keys()), 'test_cnt': list(test_cnts.values())})

    cnt_df = pd.merge(train_cnts,val_cnts,on='cat',how='left').merge(test_cnts,on='cat',how='left')

    # print(cnt_df.head())

    # Create Models
    model = models.resnet50(pretrained=True)
    # model = models.wide_resnet50_2(pretrained=True)
    # model = models.{cnn_sel}(pretrained=True)
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
        
    # # print(model)

    n_inputs = model.fc.in_features

    model.fc = nn.Sequential(
                        nn.Linear(n_inputs, 256), 
                        nn.ReLU(), 
                        nn.Dropout(0.4),
                        nn.Linear(256, n_classes),                   
                        nn.LogSoftmax(dim=1))

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')

    if train_on_gpu: 
        model = model.cuda()

    if multi_gpu:
        model = nn.DataParallel(model)
        

    if multi_gpu:
        print(model.module.fc)
    else:
        print(model.fc)
        

    model.class_to_idx = data['train'].class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    # print(list(model.idx_to_class.items())[:10])


    criterion = nn.NLLLoss() 
    optimizer = optim.Adam(model.parameters())

    # # for p in optimizer.param_groups[0]['params']:
    # #     if p.requires_grad:
    # #         print(p.shape)

    def train(model,
            criterion,
            optimizer,
            train_loader,
            valid_loader,
            save_file_name,
            max_epochs_stop=3,
            n_epochs=20,
            print_every=1):
        """Train a PyTorch Model

        Params
        --------
            model (PyTorch model): cnn to train
            criterion (PyTorch loss): objective to minimize
            optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
            train_loader (PyTorch dataloader): training dataloader to iterate through
            valid_loader (PyTorch dataloader): validation dataloader used for early stopping
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            model (PyTorch model): trained cnn with best weights
            history (DataFrame): history of train and validation loss and accuracy
        """
        # Creates a .txt file that will contain the logs
        # f = open(logs_file_name)

        # Early stopping intialization
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        valid_max_acc = 0
        history = []

        # Number of epochs already trained (if using loaded in model weights)
        try:
            print(f'Model has been trained for: {model.epochs} epochs.\n')
        except:
            model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()

        # Main loop
        for epoch in range(n_epochs):
            # Append to .txt
    #         with open(logs_file_name, 'a+') as f:
    #             print('-'*10, 'Fold:', fold+1, '-'*10, file=f)
            print('-'*10, 'Epoch:', epoch+1, '-'*10)

            #Measure how long the training epoch takes.
            start_train = timer()

            # Reset the total loss for this epoch.
            total_train_loss = 0
            total_train_acc = 0

            model.train()

            # Training loop
            for ii, (data, target) in enumerate(train_loader):
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                optimizer.zero_grad()
                # Predicted outputs are probabilities
                output = model(data)

                # Loss and backpropagation of gradients
                loss = criterion(output, target)

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item() * data.size(0)

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Update the parameters
                optimizer.step()

                # Calculate accuracy by finding max probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply accuracy times the number of examples in batch
                total_train_acc += accuracy.item()  * data.size(0)

            # Track training progress
            training_time = timer() - start_train
            print(
                f'Epoch: {epoch+1}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {training_time:.2f} seconds elapsed in epoch.',
                end='\r')
                
                # After training loops ends, start validation
    #             else:
            model.epochs += 1

            # Don't need to keep track of gradients
            start_val = timer()
            # Put the model in evaluation mode--the dropout layers behave differently  during evaluation.
            model.eval()

            # Tracking variables         
            total_valid_loss = 0 
            total_valid_acc = 0

            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():      
                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                
                # Accumulate the validation loss.
                total_valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                accuracy = torch.mean(
                    correct_tensor.type(torch.FloatTensor))
                # Multiply  accuracy times the number of examples
                total_valid_acc += accuracy.item() * data.size(0)


            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_loader.dataset)      
            avg_train_acc = total_train_acc / len(train_loader.dataset)

            # Calculate average accuracy
            avg_valid_loss = total_valid_loss / len(valid_loader.dataset)
            avg_valid_acc = total_valid_acc / len(valid_loader.dataset)
            validation_time = timer() - start_val

            history.append([epoch + 1, avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, training_time,validation_time ])

            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                str8 =f'\nEpoch: {epoch+1} \tTraining Loss: {avg_train_loss:.4f} \tValidation Loss: {avg_valid_loss:.4f}'
                flog.write(str8)
                print(str8)
                str9 = f'\t\tTraining Accuracy: {100 * avg_train_acc:.2f}%\t Validation Accuracy: {100 * avg_valid_acc:.2f}%'
                flog.write(str9)
                print(str9)

            # Save the model if validation loss decreases
            if avg_valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = avg_valid_loss
                valid_best_acc = avg_valid_acc
                best_epoch = epoch+1

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    str1=f'\nEarly Stopping! Total epochs: {epoch+1}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * avg_valid_acc:.2f}'
                    flog.write(str1)
                    print(str1)
                    total_time = timer() - overall_start

                    str2=f'\n{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    flog.write(str2)
                    print(str2)

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'epoch', 'avg_train_loss', 'avg_valid_loss', 'avg_train_acc',
                            'avg_valid_acc', 'training_time', 'validation_time'
                        ])
                    
                    return model, history


        # Attach the optimizer
        model.optimizer = optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        str3=f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * avg_valid_acc:.2f}%'
        flog.write(str3)
        print(str3)

        str4=f'\n{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
        flog.write(str4)
        print(str4)

        # Format history
        history = pd.DataFrame(
            history,
            columns=[
                            'epoch', 'avg_train_loss', 'avg_valid_loss', 'avg_train_acc',
                            'avg_valid_acc', 'training_time', 'validation_time'
                        ])

        # PRINT INFO
        # Append to .log file
        
            # with open(logs_file_name, 'a+') as f:
        str5='\nEpoch: {}/{} | avg_train_loss: {:.4} | avg_valid_loss: {:.4} | avg_train_acc: {:.3} | avg_valid_acc: {:.3} |training_time: {:.2f} seconds per epoch | validation_time: {:.2f} seconds per epoch' .\
                    format(epoch+1, n_epochs, avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, training_time, validation_time)
        flog.write(str5)  
        str6=f'\n{total_time:.2f} total seconds elapsed.'
        flog.write(str6)             
            # Print to console
        print('avg_train_loss: {:.4} | avg_valid_loss: {:.4} | avg_train_acc: {:.3} | avg_valid_acc: {:.3} | training_time: {:.2f} seconds per epoch | validation_time: {:.2f} seconds per epoch' .\
                format(avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, training_time, validation_time))

        return model, history

    model, history = train(
        model,
        criterion,
        optimizer,
        dataloaders['train'],
        dataloaders['val'],
        save_file_name=save_file_name,
        max_epochs_stop=max_epochs_stop,
        n_epochs=epochs,
        print_every=1)

    history.to_csv(result_dir+"/training_history.csv",index=0)

    plt.figure(figsize=(8, 6))
    for c in ['avg_train_loss', 'avg_valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training and Validation Losses')
    plt.savefig(f'{result_dir}/train_val_loss.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    for c in ['avg_train_acc', 'avg_valid_acc']:
        plt.plot(
            100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(f'{result_dir}/train_val_acc.png')
    plt.close()

    # Save Model
    def save_checkpoint(model, path):
        """Save a PyTorch model checkpoint

        Params
        --------
            model (PyTorch model): model to save
            path (str): location to save model. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        model_name = path.split('-')[0].split('/')[-1]
        assert (model_name in model_choices
                            ), "Path must have the correct model name"

        # Basic details
        checkpoint = {
            'class_to_idx': model.class_to_idx,
            'idx_to_class': model.idx_to_class,
            'epochs': model.epochs,
        }

            # Extract the final classifier and the state dictionary
        if model_name == 'vgg16':
            # Check to see if model was parallelized
            if multi_gpu:
                checkpoint['classifier'] = model.module.classifier
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['classifier'] = model.classifier
                checkpoint['state_dict'] = model.state_dict()

        elif model_name == 'resnet50':
            if multi_gpu:
                checkpoint['fc'] = model.module.fc
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['fc'] = model.fc
                checkpoint['state_dict'] = model.state_dict()

        elif model_name == 'squeezenet':
            if multi_gpu:
                checkpoint['fc'] = model.module.fc
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['fc'] = model.fc
                checkpoint['state_dict'] = model.state_dict()

        elif model_name == 'googlenet':
            if multi_gpu:
                checkpoint['fc'] = model.module.fc
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['fc'] = model.fc
                checkpoint['state_dict'] = model.state_dict()    

        elif model_name == 'wide_resnet50_2':
            if multi_gpu:
                checkpoint['fc'] = model.module.fc
                checkpoint['state_dict'] = model.module.state_dict()
            else:
                checkpoint['fc'] = model.fc
                checkpoint['state_dict'] = model.state_dict()    

        # Add the optimizer
        checkpoint['optimizer'] = model.optimizer
        checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

        # Save the data to the path
        torch.save(checkpoint, path)

    save_checkpoint(model, path=checkpoint_path)
    
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

        # Get the model name
        model_name = path.split('-')[0].split('/')[-1]
        assert (model_name in model_choices), "Path must have the correct model name"

        # Load in checkpoint
        checkpoint = torch.load(path)

        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = checkpoint['classifier']

        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.fc = checkpoint['fc']

        elif model_name == 'squeezenet':
            model = models.squeezenet(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.fc = checkpoint['fc']

        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=True)
            # Make sure to set parameters as not trainable
            for param in model.parameters():
                param.requires_grad = False
            model.fc = checkpoint['fc']

        elif model_name == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=True)
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

        # Move to gpu
        if multi_gpu:
            model = nn.DataParallel(model)

        if train_on_gpu:
            model = model.to('cuda')

        # Model basics
        model.class_to_idx = checkpoint['class_to_idx']
        model.idx_to_class = checkpoint['idx_to_class']
        model.epochs = checkpoint['epochs']

        # Optimizer
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model, optimizer

    model, optimizer = load_checkpoint(path=checkpoint_path)

    # Inference
#     def process_image(image_path):
#         """Process an image path into a PyTorch tensor"""

#         image = Image.open(image_path)
#         # Resize
#         img = image.resize((256, 256))

#         # Center crop
#         width = 256
#         height = 256
#         new_width = 224
#         new_height = 224

#         left = (width - new_width) / 2
#         top = (height - new_height) / 2
#         right = (width + new_width) / 2
#         bottom = (height + new_height) / 2
#         img = img.crop((left, top, right, bottom))

#         # Convert to numpy, transpose color dimension and normalize
#         img = np.array(img).transpose((2, 0, 1)) / 256

#         # Standardization
#         means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
#         stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

#         img = img - means
#         img = img / stds

#         img_tensor = torch.Tensor(img)

#         return img_tensor

#     def predict(image_path, model, topk=1):
#         """Make a prediction for an image using a trained model

#         Params
#         --------
#             image_path (str): filename of the image
#             model (PyTorch model): trained model for inference
#             topk (int): number of top predictions to return

#         Returns
            
#         """
#         real_class = image_path.split('/')[-2]

#         # Convert to pytorch tensor
#         img_tensor = process_image(image_path)

#         # Resize
#         if train_on_gpu:
#             img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
#         else:
#             img_tensor = img_tensor.view(1, 3, 224, 224)

#         # Set to evaluation
#         with torch.no_grad():
#             model.eval()
#             # Model outputs log probabilities
#             out = model(img_tensor)
#             ps = torch.exp(out)

#             # Find the topk predictions
#             topk, topclass = ps.topk(topk, dim=1)

#             # Extract the actual classes and probabilities
#             top_classes = [
#                 model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
#             ]
#             top_p = topk.cpu().numpy()[0]

#             return img_tensor.cpu().squeeze(), top_p, top_classes, real_class

#     np.random.seed = 100

#     def random_test_image():
#         """Pick a random test image from the test directory"""
#         c = np.random.choice(categories)
#         root = testdir +"/"+ c + '/'
#         img_path = root + np.random.choice(os.listdir(root))
# #         print(img_path)
#         return img_path

#     _ = imshow_tensor(process_image(random_test_image()))

#     img, top_p, top_classes, real_class = predict(random_test_image(), model,topk=1)

#     print('top_probability:', top_p, 'pred class:', top_classes, 'actual class:', real_class)

    # Display Preds
#     def display_prediction(image_path, model, topk):
#         """Display image and preditions from model"""

#         # Get predictions
#         img, ps, classes, y_obs = predict(image_path, model, topk)
#         # Convert results to dataframe for plotting
#         result = pd.DataFrame({'p': ps}, index=classes)

#         # Show the image
#         # plt.figure(figsize=(16, 5))
#         # ax = plt.subplot(1, 2, 1)
#         # ax, img = imshow_tensor(img, ax=ax)
#         img = imshow_tensor(img)

#         # Set title to be the actual class
#         # ax.set_title(y_obs, size=20)
#         plt.title(f'Actual class= {y_obs}')
        
#         # ax = plt.subplot(1, 2, 2)
#         # Plot a bar plot of predictions
#         # result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
#         # plt.xlabel('Predicted Probability')
#         plt.tight_layout()
#         plt.savefig(f"{result_dir}/pred_img.png")
#         plt.close()

#     display_prediction(random_test_image(), model, topk=1)


    def accuracy(output, target, topk=(1, )):
    # def accuracy(output, target):    
        """Compute the topk accuracy(s)"""
        # k=1
        if train_on_gpu:
            output = output.to('cuda')
            target = target.to('cuda')

        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            # print('maxk',maxk)
            # Find the predicted classes and transpose
            _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()

            # Determine predictions equal to the targets
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []

            # For each k, find the percentage of correct
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res

    testiter = iter(dataloaders['test'])
    # Get a batch of testing images and labels
    test_features, test_targets = next(testiter)

    if train_on_gpu:
        test_res = accuracy(model(test_features.to('cuda')), test_targets, topk=(1, 2))
    else:
        test_res = accuracy(model(test_features), test_targets, topk=(1, 2))
#     print('Test accuracy', test_res)

    # Function to Evaluate Model Over All Classes
    def evaluate(model, test_loader, criterion, topk=(1, 2)):
        """Measure the performance of a trained PyTorch model

        Params
        --------
            model (PyTorch model): trained cnn for inference
            test_loader (PyTorch DataLoader): test dataloader
            topk (tuple of ints): accuracy to measure

        Returns
        --------
            results (DataFrame): results for each category

        """
        confusion_matrix = torch.zeros(n_classes, n_classes)
        classes = []
        losses = []
        # Hold accuracy results
        acc_results = np.zeros((len(test_loader.dataset), len(topk)))
        i = 0

        model.eval()
        with torch.no_grad():

            # Testing loop
            for data, targets in test_loader:

                # Tensors to gpu
                if train_on_gpu:
                    data, targets = data.to('cuda'), targets.to('cuda')

                # Raw model output
                out = model(data)
                _, predscm = torch.max(out, 1)
                for t, p in zip(targets.view(-1), predscm.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                
                # Iterate through each example
                for pred, true in zip(out, targets):
                    # Find topk accuracy
                    acc_results[i, :] = accuracy(
                        pred.unsqueeze(0), true.unsqueeze(0), topk)
                    classes.append(model.idx_to_class[true.item()])
                    # Calculate the loss
                    loss = criterion(pred.view(1, n_classes), true.view(1))
                    losses.append(loss.item())
                    i += 1

        # Send results to a dataframe and calculate average across classes
        results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
        # results['img_file']=ImageFile
        results['class'] = classes
        results['loss'] = losses
        results = results.groupby(classes).mean()
        
        return results.reset_index().rename(columns={'index': 'class'}),confusion_matrix

    criterion = nn.NLLLoss()
    # Evaluate the model on all the training data
    test_results,confusion_matrix = evaluate(model, dataloaders['test'], criterion)

    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else '.0f'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f"{result_dir}/test_confusion_matrix.png")
        plt.close()
        return plt


    classnames = [model.idx_to_class[i] for i in range(0,n_classes)]
    plt.figure(figsize=(20,20))
    plt = plot_confusion_matrix(confusion_matrix, classnames)
    plt.close()

    def cm_as_dataframe(cm):
        cm = pd.DataFrame(cm)
        cm.columns = classnames
        cm.index = classnames
        cm = cm.reset_index()
        return cm

    # cm_as_dataframe(confusion_matrix.cpu().numpy()).to_csv(f'../reports/{jobname}_confmat.csv',index=None)

    overall_test_acc = (confusion_matrix.diag().sum()/confusion_matrix.sum()) * 100
    str7 = f"Overall Test Accuracy:{overall_test_acc:.2f}"
    flog.write(str7)
    print(str7)
    
    test_results = test_results.merge(cnt_df,left_on='class',right_on='cat')

    #save test prediction result
    # Defining Labels and Predictions
    test_truelabels = []
    test_predictions = []
#     test_onnx_preds=[]
    idx_to_class = model.idx_to_class
    model.eval()
    with torch.no_grad():
        print("Getting predictions from test set...")
        for data, targets in dataloaders['test']:
            # Tensors to gpu
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')           
            for label in targets.cpu().numpy():
                test_truelabels.extend([idx_to_class[label]])
            test_output = model(data)
            _, test_pred = torch.max(test_output, dim=1)
            test_predictions.extend([idx_to_class[val] for val in test_pred.cpu().numpy()])
    test_pred_df = pd.DataFrame({'pred':test_predictions,'actual':test_truelabels} )   
    print(len(test_pred_df))     
    # print(test_pred_df.head())
    test_pred_df.to_csv(f'{result_dir}/test_pred.csv',index=None)
    print(f"Prediction output of test images saved to {result_dir}/test_pred.csv")
    test_results.to_csv(f'{result_dir}/test_accuracy.csv',index=None)
    print(f"Test accuracy table saved to {result_dir}/test_accuracy.csv")
            
   ####
    #Refactored code to export pytorch model as onnx format for inference
    #https://github.com/prabhat00155/onnx-odsc-tutorial/blob/master/pytorch%20experiment.ipynb
    #####
#     for test_features, test_targets in dataloaders['test']:     
#         dummy_input = Variable(torch.randn(test_features.size()[0], 3, img_size, img_size)).to('cuda')   
#         torch.onnx.export(model, dummy_input, f"{result_dir}/resnet50_imgSize224_test.onnx")
#         onnx_test_model = f"{result_dir}/resnet50_imgSize224_test.onnx"
#         session  = onnxruntime.InferenceSession(onnx_test_model,None)
#         input_name = session.get_inputs()[0].name
#         output_name = session.get_outputs()[0].name
#         result = session.run([output_name], {input_name: test_features.numpy()})
#         onnx_output=np.argmax(result[0], axis=1)
#         test_onnx_preds.extend([idx_to_class[val]for val in onnx_output])
#         test_truelabels.extend([idx_to_class[label]for label in test_targets.cpu().numpy()])   
    
#     test_pred_df = pd.DataFrame({'pred':test_onnx_preds,'actual':test_truelabels} )   
#     # print(len(test_pred_df))     
#     # print(test_pred_df.head())
#     test_pred_df.to_csv(f'{result_dir}/test_pred.csv',index=None)
#     print(f"Prediction output of test images saved to {result_dir}/test_pred.csv")
#     test_results.to_csv(f'{result_dir}/test_accuracy.csv',index=None)
#     print(f"Test accuracy table saved to {result_dir}/test_accuracy.csv")

#     ##### save all image prediction result
#     full_unaug_truelabels = []
#     full_unaug_predictions = []
        
#     model.eval()
#     with torch.no_grad():
#         print("Getting predictions from full unaug set...")
#         for data, targets in dataloaders['full_unaug']:
#             # Tensors to gpu
#             if train_on_gpu:
#                 data, targets = data.to('cuda'), targets.to('cuda')           
#             for label in targets.cpu().numpy():
#                 full_unaug_truelabels.extend([idx_to_class[label]])
#             full_unaug_output = model(data)
#             _, full_unaug_pred = torch.max(full_unaug_output, dim=1)
#             full_unaug_predictions.extend([idx_to_class[val] for val in full_unaug_pred.cpu().numpy()])
#     full_unaug_pred_df = pd.DataFrame({'pred':full_unaug_predictions,'actual':full_unaug_truelabels} ) 
#     print(len(full_unaug_pred_df))
#     full_unaug_pred_df.to_csv(f'{result_dir}/all_pred.csv',index=None)
#     print(f"Prediction output of all images saved to {result_dir}/all_pred.csv")
    
    
# #     full_unaug_label = full_unaug_pred_df['actual']
# #     full_unaug_pred_pred_cat = full_unaug_pred_df['pred']
# #     cm=confusion_matrix(full_unaug_label,full_unaug_pred_pred_cat) 
# #     accuracy = np.trace(cm) / np.sum(cm).astype('float') 
# #     misclass = 1 - accuracy
# #     cmap = plt.get_cmap('Blues') 
# #     normalize = False 
# #     plt.figure(figsize=(8, 6)) 
# #     plt.imshow(cm, interpolation='nearest', cmap=cmap) 
# #     plt.title('Full unaug conf mat') 
# #     plt.colorbar()
# #     tick_marks = np.arange(n_classes)
# #     plt.xticks(tick_marks, classnames, rotation=45)
# #     plt.yticks(tick_marks, classnames)
# #     plt.savefig(f"{result_dir}/test_confusion_matrix.png")
    
#     ##### save all blind test image prediction result
#     blind_test_truelabels = []
#     blind_test_predictions = []
        
#     model.eval()
#     with torch.no_grad():
#         print("Getting predictions from blind test set...")
#         for data, targets in dataloaders['blind_test']:
#             # Tensors to gpu
#             if train_on_gpu:
#                 data, targets = data.to('cuda'), targets.to('cuda')           
#             for label in targets.cpu().numpy():
# #                 print(label)
#                 blind_test_truelabels.extend([idx_to_class[label]])
#             test_output = model(data)
#             _, test_pred = torch.max(test_output, dim=1)
#             blind_test_predictions.extend([idx_to_class[val] for val in test_pred.cpu().numpy()])
# #             break
#     blind_test_pred_df = pd.DataFrame({'pred':blind_test_predictions,'actual':blind_test_truelabels} ) 
#     blind_test_pred_df.to_csv(f'{result_dir}/blind_test_red.csv',index=None)   
#     print(len(blind_test_pred_df))
#     print(f"Prediction output of all images saved to {result_dir}/blind_test_pred.csv")