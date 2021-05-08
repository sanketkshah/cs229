#!/usr/bin/env python
# coding: utf-8

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
import pandas as pd
import pickle
import time
import math
import os
import copy
from imagenetv2_pytorch import ImageNetV2Dataset
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from pytorch_metric_learning import losses
import wandb
import argparse
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Batch size for training (change depending on how much memory you have)
BATCH_SIZE = 256

# Number of epochs to train for 
EPOCHS = 500

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using GPU? {torch.cuda.is_available()}")

# ## Training Functions

def train_model(model, dataloaders, criterion, optimizer, label_embeddings, num_epochs=25, is_cv=False, embedding_to_label=None, patience=50):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    time_since_best = 0
    

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
            num_samples_seen = 0

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
                    outputs = model(inputs)      
                    outputs = outputs / outputs.norm(dim=-1, keepdim=True)
                    if is_cv:
                        loss = criterion(outputs, labels)
                    else:
                        loss = criterion(outputs, labels, label_embeddings)

                    if embedding_to_label is not None and is_cv is not True:
                        preds = embedding_to_label(outputs, label_embeddings)
                    else:
                        _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_samples_seen += inputs.size(0)

            epoch_loss = running_loss / num_samples_seen
            epoch_acc = running_corrects.double() / num_samples_seen

            if phase == 'train':
                wandb.log({'Epoch': epoch, 'Train Loss': epoch_loss, 'Train Accuracy': epoch_acc})
            else:
                wandb.log({'Epoch': epoch, 'Val Loss': epoch_loss, 'Val Accuracy': epoch_acc})

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            # deep copy the model if it's good
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), f'/scratch/{wandb.run.name}_{epoch_acc}.mdl')
                torch.save(model.state_dict(), f'./models/{wandb.run.name}_final.mdl')
                time_since_best = 0

        if time_since_best > patience:
            break
        time_since_best += 1

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, embedding_dim, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if "resnet" in model_name:
        """ Resnet18
        """
        if model_name == "beefyresnet":
            num_layers = 3
        elif model_name == "verybeefyresnet":
            num_layers = 5
        else:
            num_layers = 1

        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features

        if num_layers > 1:
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            for _ in range(num_layers - 2):
                model_ft = nn.Sequential(
                    model_ft,
                    nn.ReLU(),
                    nn.Linear(num_ftrs, num_ftrs)
                )
            model_ft = nn.Sequential(
                    model_ft,
                    nn.ReLU(),
                    nn.Linear(num_ftrs, embedding_dim)
                )
        else:
            model_ft.fc = nn.Linear(num_ftrs, embedding_dim)
            
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        sys.exit(0)
    
    return model_ft, input_size


# Data augmentation and normalization for training, just normalization for validation
def create_dataloaders(dataset_name, data_transforms, input_size, batch_size):
    dataloaders_dict = {}

    print("Initializing Datasets and Dataloaders...")
    if dataset_name == "imagenetv2":
        # Create training and validation datasets
        train_dataset = ImageNetV2Dataset(transform=data_transforms['train'])
        test_dataset = ImageNetV2Dataset(transform=data_transforms['val'])

        train_test_splits_file = 'split_indices.pkl'

        if os.path.exists(train_test_splits_file):
            indices_split = pickle.load(open(train_test_splits_file, 'rb'))
        else:
            index_to_class = {idx: cl for idx, (_, cl) in enumerate(train_dataset)}
            class_to_index = {idx: [] for idx in range(1000)}
            for idx, cl in index_to_class.items():
                class_to_index[cl].append(idx)

            indices_split = {'train': [], 'val': [], 'test': []}
            for cl in class_to_index:
                shuffle(class_to_index[cl])
                indices_split['train'].extend(class_to_index[cl][:int(0.7 * len(class_to_index[cl]))])
                indices_split['val'].extend(class_to_index[cl][int(0.7 * len(class_to_index[cl])):int(0.9 * len(class_to_index[cl]))])
                indices_split['test'].extend(class_to_index[cl][int(0.9 * len(class_to_index[cl])):])
            
            pickle.dump(indices_split, open(train_test_splits_file, 'wb'))
            

        # Create training and validation dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(Subset(test_dataset,indices_split[x]), batch_size=batch_size, shuffle=True, num_workers=4) for x in ['val', 'test']}
        dataloaders_dict['train'] = torch.utils.data.DataLoader(Subset(train_dataset, indices_split['train']), batch_size=batch_size, shuffle=True, num_workers=4)

    elif dataset_name == "imagenetv2cifar100":
        # Create training and validation datasets
        print("Custom imagenetv2cifar dataset")
        train_dataset = ImageNetV2Dataset(transform=data_transforms['train']['imagenetv2'])
        test_dataset = ImageNetV2Dataset(transform=data_transforms['val']['imagenetv2'])

        train_test_splits_file = 'split_indices.pkl'

        if os.path.exists(train_test_splits_file):
            indices_split = pickle.load(open(train_test_splits_file, 'rb'))
        else:
            index_to_class = {idx: cl for idx, (_, cl) in enumerate(train_dataset)}
            class_to_index = {idx: [] for idx in range(1000)}
            for idx, cl in index_to_class.items():
                class_to_index[cl].append(idx)

            indices_split = {'train': [], 'val': [], 'test': []}
            for cl in class_to_index:
                shuffle(class_to_index[cl])
                indices_split['train'].extend(class_to_index[cl][:int(0.7 * len(class_to_index[cl]))])
                indices_split['val'].extend(class_to_index[cl][int(0.7 * len(class_to_index[cl])):int(0.9 * len(class_to_index[cl]))])
                indices_split['test'].extend(class_to_index[cl][int(0.9 * len(class_to_index[cl])):])
            
            pickle.dump(indices_split, open(train_test_splits_file, 'wb'))
            

        train_dataset1 = Subset(train_dataset, indices_split['train'])
        test_dataset1 = Subset(test_dataset, indices_split['val'])

        train_dataset2 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transforms['train']['cifar100'])
        test_dataset2 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transforms['val']['cifar100'])

        train_dataset2.targets = [x + 1000 for x in train_dataset2.targets]
        test_dataset2.targets = [x + 1000 for x in test_dataset2.targets]

        final_train_dataset = torch.utils.data.ConcatDataset([train_dataset1, train_dataset2])
        final_test_dataset = torch.utils.data.ConcatDataset([test_dataset1, test_dataset2])

        # Create training and validation dataloaders
        dataloaders_dict['train'] = torch.utils.data.DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloaders_dict['val'] = torch.utils.data.DataLoader(final_test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    else:
        if dataset_name == "imagenet":
            train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', download=True, transform=data_transforms['train'])
            test_dataset = torchvision.datasets.ImageNet(root='./data', split='val', download=True, transform=data_transforms['val'])
        elif dataset_name == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=data_transforms['train'])
            test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=data_transforms['val'])
        elif dataset_name == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=data_transforms['train'])
            test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=data_transforms['val'])
        else:
            print("Invalid dataset name, exiting...")
            sys.exit(0)

        dataloaders_dict['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        dataloaders_dict['val'] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloaders_dict

def generate_label_embeddings(embedding_type, embedding_dim, dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if embedding_type == "onehot":
        # One-hot text embeddings
        label_embeddings = torch.from_numpy(np.eye(embedding_dim, dtype=np.float32)).to(device)
    else:
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        # Load labeltext
        
        if dataset_name == "imagenet" or dataset_name == "imagenetv2" or dataset_name == "imagenetv2cifar100":
            classidx_to_label = pickle.load(open('classidx_to_label.pkl', 'rb'))
            classidx_to_label2 = pickle.load(open('./data/cifar-100-python/meta', 'rb'))['fine_label_names']
        elif dataset_name == "cifar10":
            classidx_to_label = pickle.load(open('./data/cifar-10-batches-py/batches.meta', 'rb'))['label_names']
        elif dataset_name == "cifar100":
            classidx_to_label = pickle.load(open('./data/cifar-100-python/meta', 'rb'))['fine_label_names']
        elif dataset_name == "cifar100super":
            classidx_to_label = pickle.load(open('./data/cifar-100-python/meta', 'rb'))['coarse_label_names']
        else:
            print("ERROR: Dataset without labels")
            sys.exit(0)

        if embedding_type == "label":
            # Text embeddings based on class label text
            if dataset_name == "imagenet" or dataset_name == "imagenetv2":
                labels = list(classidx_to_label.values())
            elif dataset_name == "imagenetv2cifar100":
                labels = list(classidx_to_label.values()) + list(classidx_to_label2)
            else:
                labels = list(classidx_to_label)
                
        elif embedding_type == 'wiki':
            if dataset_name == "imagenet" or dataset_name == "imagenetv2":
                # Load wikitext
                wiki_path = 'ImageNet-Wiki_dataset/class_article_text_descriptions/class_article_text_descriptions_trainval.pkl'
                wiki_articles = pickle.load(open(wiki_path, 'rb'))
                wiki_label_map = pd.read_csv('LOC_synset_mapping.txt', sep=': ', names=['wnid', 'labels'])


                wiki_labels = {}
                for i in classidx_to_label.keys():
                    wiki_labels[wiki_label_map.iloc[i]['wnid']] = classidx_to_label[i]

                for i in wiki_articles.keys():
                    try:
                        wiki_labels[wiki_articles[i]['wnid']] = wiki_articles[i]['articles'][0]
                    except:
                        pass
                labels = list(wiki_labels.values())
            else:
                sys.exit(0)
        elif embedding_type == 'clip':
            # Text embeddings based on class label text
            clip_labels={}
            if dataset_name == "imagenet" or dataset_name == "imagenetv2" or dataset_name == "imagenetv2cifar100":
                for i in classidx_to_label.keys():
                    clip_labels[i] = 'A photo of ' + str(classidx_to_label[i].split(",")[0].rstrip().lstrip())

                if dataset_name == "imagenetv2cifar100":
                    for i in range(len(classidx_to_label2)):
                        clip_labels[i + 1000] = 'A photo of ' + str(classidx_to_label2[i].split(",")[0].rstrip().lstrip())

            else:
                for i in range(len(classidx_to_label)):
                    clip_labels[i] = 'A photo of ' + str(classidx_to_label[i].split(",")[0].rstrip().lstrip())
            labels = list(clip_labels.values())

        label_embeddings = torch.from_numpy(model.encode(labels)).to(device)
        label_embeddings = label_embeddings / label_embeddings.norm(dim=-1, keepdim=True)

    return label_embeddings
    
# Setup the loss fxn
def cliploss(image_embeddings, labels, label_embeddings, temperature=1.0):
    text_embeddings = label_embeddings[labels]
    logits = (text_embeddings @ image_embeddings.T) * temperature
    targets = torch.arange(labels.shape[0]).to(device)

    texts_loss = nn.CrossEntropyLoss()(logits, targets)
    images_loss = nn.CrossEntropyLoss()(logits.T, targets)
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    return loss


def cliplossv2(image_embeddings, labels, label_embeddings, temperature=1.0):
    unique_labels, map_to_unique = labels.unique(return_inverse=True)

    text_embeddings = label_embeddings[unique_labels]
    logits = (image_embeddings @ text_embeddings.T) * temperature

    loss = nn.CrossEntropyLoss()(logits, map_to_unique)
    return loss


def cosineloss(image_embeddings, labels, label_embeddings):
    text_embeddings = label_embeddings[labels]
    loss = -torch.sum(image_embeddings * text_embeddings, dim=-1)
    return loss.mean()


def closest_labelembedding(image_embeddings, label_embeddings):
    return (image_embeddings @ label_embeddings.T).argmax(-1)



## Testing the model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, type=str, choices=['imagenet', 'imagenetv2', 'cifar10', 'cifar100', 'imagenetv2cifar100'], help="dataset to use")
    parser.add_argument("-e", "--embedding_type", required=True, type=str, choices=['onehot', 'label', 'wiki', 'clip'], help="text embeddings to use")
    parser.add_argument("-l", "--loss", required=True, type=str, choices=['cliplossv2', 'cliploss', 'cosineloss', 'xeloss'], help="loss to use")
    parser.add_argument("-o", "--optim", required=True, type=str, choices=['adam', 'sgd'], help="optimizer to use")
    parser.add_argument("-n", "--network", required=True, type=str, choices=['resnet', 'beefyresnet', 'verybeefyresnet'], help="type of network")


    args = parser.parse_args()

    dataset_name = args.dataset
    dataset_label_sizes = {'imagenet':1000, 'imagenetv2': 1000, 'cifar10': 10, 'cifar100': 100, 'imagenetv2cifar100': 1100}
    pretrain_flag = {'imagenet':True, 'imagenetv2': True, 'cifar10': True, 'cifar100': True, 'imagenetv2cifar100': True}
    feature_extract_flag = {'imagenet':True, 'imagenetv2': True, 'cifar10': True, 'cifar100': True, 'imagenetv2cifar100': True}

    mean = {'cifar10': (0.4914, 0.4822, 0.4465), 'cifar100': (0.5071, 0.4867, 0.4408), 'imagenet':(0.485, 0.456, 0.406), 'imagenetv2': (0.485, 0.456, 0.406)}
    std  = {'cifar10': (0.2023, 0.1994, 0.2010), 'cifar100': (0.2675, 0.2565, 0.2761), 'imagenet':(0.229, 0.224, 0.225), 'imagenetv2': (0.229, 0.224, 0.225)}


    model_name = args.network
    embedding_type = args.embedding_type

    # Number of classes in the dataset
    if embedding_type == "onehot":
        embedding_dim = dataset_label_sizes[dataset_name]
    else:
        embedding_dim = 768

    # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, embedding_dim, feature_extract_flag[dataset_name], use_pretrained=pretrain_flag[dataset_name])
    print(model_ft)

    # Send the model to GPU
    model_ft = model_ft.to(device)

    if dataset_name == "imagenetv2cifar100":
        data_transforms = {
            'train': {
                'imagenetv2': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean['imagenetv2'], std['imagenetv2']),
                ]),
                'cifar100': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean['cifar100'], std['cifar100']),
                ])
            },

            'val': {
                'imagenetv2': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean['imagenetv2'], std['imagenetv2'])
                ]),
                'cifar100': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean['cifar100'], std['cifar100'])
                ])
            }
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean[dataset_name], std[dataset_name])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean[dataset_name], std[dataset_name])
            ]),
        }

    dataloaders_dict = create_dataloaders(dataset_name, data_transforms, input_size, BATCH_SIZE)

    label_embeddings = generate_label_embeddings(embedding_type, embedding_dim, dataset_name)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract_flag[dataset_name]:
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
    if args.optim == "adam":
        optimizer_ft = optim.Adam(params_to_update)
    else:
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)


    # # Train and evaluate

    # Run Training and Validation Step
    # --------------------------------
    # 
    # Finally, the last step is to setup the loss for the model, then run the
    # training and validation function for the set number of epochs. Notice,
    # depending on the number of epochs this step may take a while on a CPU.
    # Also, the default learning rate is not optimal for all of the models, so
    # to achieve maximum accuracy it would be necessary to tune for each model
    # separately.

    loss_types = {'cliploss':cliploss, 'cliplossv2':cliplossv2, 'cosineloss':cosineloss, 'xeloss': nn.CrossEntropyLoss()}
    criterion = loss_types[args.loss]


    # Initialise WandB
    run = wandb.init(
        project="cs229-finetuning",
        config={
            "loss": args.loss,
            "model_name": model_name,
            'dataset': dataset_name,
            'embedding_type': embedding_type,
            'batch_size': BATCH_SIZE,
            'optimizer': args.optim
        })
    wandb.run.name = f"{args.loss}_{embedding_type}_{model_name}_{dataset_name}_{BATCH_SIZE}_{args.optim}"
    print(wandb.config)
    wandb.watch(model_ft)

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, label_embeddings, num_epochs=EPOCHS, is_cv=(args.loss=="xeloss"), embedding_to_label=closest_labelembedding)


    torch.save(model_ft.state_dict(), f'./models/{wandb.run.name}_final.mdl')
    run.finish()

if __name__ == '__main__':
    main()
