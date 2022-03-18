"""
Created on Sat Mar 12 09:45:01 2022
@author: Laurent Ferhi

Training flower species classifier
"""

# Imports here
import numpy as np
import PIL

import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim

from collections import OrderedDict
import argparse
import json

### Argument parser

# Arguments definition
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help = 'Specify directory where training data is located (required)', type = str)
parser.add_argument('--save_dir', help = 'Set directory to save checkpoints', default='', type = str)
parser.add_argument('--arch', help = 'Choose model architecture among [vgg13, vgg16] (default model vgg16)', default='vgg16',type = str)
parser.add_argument('--lrn', help = 'Define learning rate (default value is 0.001)', default=0.001, type = float)
parser.add_argument('--hidden_units', help = 'Specify hidden units in the classifier (default value is 4096)', default=4096,type = int)
parser.add_argument('--epochs', help = 'Specify number of epochs (default valus is 5)', default=5, type = int)
parser.add_argument('--gpu', help = "Use GPU for training", default=False, action='store_true')

args = parser.parse_args()

### Definition of functions

def create_data_loaders(data_dir):
    ''' Load and transform data into torchvision dataloaders
    '''
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])


    test_valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return train_data, trainloader, valid_data, validloader, test_data, testloader

def define_model(arch, hidden_units, device):
    ''' Create model architecture with transfer learning
    '''
    
    # Load classe names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    cat_number = len(cat_to_name)
    
    # Load model
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.vgg13(pretrained=True)
   
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
   # Update classifier last layer to fit categories number
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units, bias=True)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, cat_number, bias=True)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier 
    model.to(device)
    
    return model


def validation(model, dataloader, criterion):
    ''' Model validation function
    '''
    
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(dataloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_classifier(model, arch, lr, epochs, save_dir, train_data, trainloader, validloader, testloader, device):
    ''' Train, evaluate and save model
    '''
    
    # Training on the training set
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    
    print_every = 20
    steps = 0
    
    print('Training classifier, please wait...')
     
    for e in range(epochs):
        running_loss = 0

        for ii, (inputs, labels) in enumerate(trainloader):

            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model.forward(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Print stats
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                      "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))

                running_loss = 0
                model.train()
                
    # Testing network
    print('Testing network on test set...')
    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy achieved on test images is: %d%%' % (100*correct/total))
    
    # Save the checkpoint 
    model.to('cpu') 
    model.class_to_idx = train_data.class_to_idx 

    checkpoint = {
        'arch':arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }

    torch.save(checkpoint, save_dir)
    print('Model successfully saved to:',save_dir)
    return model

### MAIN
if __name__ == "__main__":
    
    ### Path to directories
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    ### Choose gpu or cpu device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    ### Get save directory
    if args.save_dir:
        var_save_dir = args.save_dir + '/checkpoint.pth'
    else:
        var_save_dir = 'checkpoint.pth'
    
    ### Display user parameters
    print('data_dir: {}\narch: {}\nhidden_units: {}\nlearning rate: {}\nepochs: {}\ndevice: {}\nsave_dir: {}'.format(
        data_dir, args.arch, args.hidden_units, args.lrn, args.epochs, device, var_save_dir
    ))

    ### launch classifier training
    train_data, trainloader, valid_data, validloader, test_data, testloader = create_data_loaders(data_dir)
    model = define_model(args.arch, args.hidden_units, device)
    train_classifier(model,args.arch, args.lrn, args.epochs, var_save_dir, train_data, trainloader, validloader, testloader, device)
    