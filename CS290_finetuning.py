#!/usr/bin/env python
# coding: utf-8

# In[9]:


# !pip install git+https://github.com/modestyachts/ImageNetV2_pytorch


# 
# Finetuning Torchvision Models
# =============================
# 
# **Author:** `Nathan Inkawhich <https://github.com/inkawhich>`__
# 
# 
# 

# In[41]:


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from random import shuffle
import time
import os
import copy
from imagenetv2_pytorch import ImageNetV2Dataset
import wandb

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# ## Parameters

# In[49]:


# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
# data_dir = "./data/imagenetv2"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 1000

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for 
num_epochs = 250

# Flag for feature extracting. When False, we finetune the whole model, 
#   when True we only update the reshaped layer params
feature_extract = False


# ## Training Functions

# In[43]:


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Initialise WandB
    run = wandb.init(
        project="cs229-finetuning",
        config={
            "model_name":model_name,
            'dataset':'imagenetv2',
            'batch_size':batch_size,
            'optimizer':optimizer
        })
    wandb.run.name = f"{wandb.config.model_name}_{wandb.config.dataset}_{wandb.config.optimizer}"
    print(wandb.config)
    wandb.watch(model)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                wandb.log({'Epoch': epoch, 'Train Loss': epoch_loss, 'Train Accuracy': epoch_acc})
            else:
                wandb.log({'Epoch': epoch, 'Val Loss': epoch_loss, 'Val Accuracy': epoch_acc})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_ft.state_dict(), f'./models/imagenetv2_{model_name}_{epoch_acc}.mdl')
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    run.finish()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# In[44]:


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


# In[50]:


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, num_layers=1):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet" or model_name == "beefyresnetwithdropout":
        """ Resnet18
        """
        if model_name == "beefyresnetwithdropout":
            num_layers = 3

        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if num_layers > 1:
#             model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            for _ in range(num_layers - 1):
                model_ft = nn.Sequential(
                    model_ft,
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Linear(num_classes, num_classes)
                )
            model_ft = nn.Sequential(
                    model_ft,
                    nn.Dropout(),
                    nn.ReLU(),
                    nn.Linear(num_classes, num_classes)
                )
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


# ## Load Data
# 
# 
# 

# In[47]:


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
train_dataset = ImageNetV2Dataset(transform=data_transforms['train'])
test_dataset = ImageNetV2Dataset(transform=data_transforms['val'])

index_to_class = {idx: cl for idx, (_, cl) in enumerate(test_dataset)}
class_to_index = {idx: [] for idx in range(1000)}
for idx, cl in index_to_class.items():
    class_to_index[cl].append(idx)

indices_split = {'train': [], 'val': [], 'test': []}
for cl in class_to_index:
    shuffle(class_to_index[cl])
    indices_split['train'].extend(class_to_index[cl][:int(0.7 * len(class_to_index[cl]))])
    indices_split['val'].extend(class_to_index[cl][int(0.7 * len(class_to_index[cl])):int(0.9 * len(class_to_index[cl]))])
    indices_split['test'].extend(class_to_index[cl][int(0.9 * len(class_to_index[cl])):])

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(Subset(test_dataset,indices_split[x]), batch_size=batch_size, shuffle=True, num_workers=4) for x in ['val', 'test']}
dataloaders_dict['train'] = torch.utils.data.DataLoader(Subset(train_dataset, indices_split['train']), batch_size=batch_size, shuffle=True, num_workers=4)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using GPU? {torch.cuda.is_available()}")


# ## Create the Optimizer
# 
# 
# 

# In[51]:


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are 
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# optimizer_ft = optim.Adam(params_to_update)


# Run Training and Validation Step
# --------------------------------
# 
# Finally, the last step is to setup the loss for the model, then run the
# training and validation function for the set number of epochs. Notice,
# depending on the number of epochs this step may take a while on a CPU.
# Also, the default learning rate is not optimal for all of the models, so
# to achieve maximum accuracy it would be necessary to tune for each model
# separately.
# 
# 
# 

# In[ ]:


# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))


# In[ ]:


torch.save(model_ft.state_dict(), f'./imagenetv2_{model_name}_{time.ctime()}.mdl')


# ## Testing the model
