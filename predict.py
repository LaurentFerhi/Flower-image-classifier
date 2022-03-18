"""
Created on Sat Mar 12 09:51:01 2022
@author: Laurent Ferhi

Predict flower species from picture
"""

import PIL
import numpy as np
from collections import OrderedDict
import torch
from torchvision import models
import json
import argparse
import pandas as pd

### Argument parser

# Arguments definition
parser = argparse.ArgumentParser()

parser.add_argument('image_dir', help = 'Specify path to image (required)', type = str)
parser.add_argument('checkpoint_dir', help = 'Specify path to model checkpoint without .pth (required)', type = str)
parser.add_argument('--top_k', help = 'Specify top k most likely classes (default valus is 5)', default=5, type = int)
parser.add_argument('--category_names', help = 'Specify mapping of categories to real names (.json file)', default='cat_to_name.json',type = str)
parser.add_argument('--gpu', help = "Use GPU for training", default=False, action='store_true')

args = parser.parse_args()

### Definition of functions

def loading_model(file_path):
    ''' Load chekpoint
    '''
    
    checkpoint = torch.load(file_path) 
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)
    else:
        model = models.vgg13(pretrained = True)
        
    model.classifier = checkpoint['classifier']
    model.load_state_dict (checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters(): 
        param.requires_grad = False 
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    im = PIL.Image.open(image)
    width, height = im.size
    
    # make thumbnail according to higher size
    if width <= height: 
        im.thumbnail(size=[256, 256*2])
    else: 
        im.thumbnail(size=[256*2, 256])
        
    # Update size and crop to 224x224
    new_width, new_height = im.size

    left = (new_width - 224)/2 
    right = left + 224 
    up = (new_height - 224)/2
    down = up + 224
    im = im.crop((left, up, right, down))
    
    # Color channels of images
    np_image = np.array(im)/255
    np_image = (np_image-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    np_image= np_image.transpose((2,0,1))

    return np_image

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device) 
    im = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor).unsqueeze(dim = 0)
        
    with torch.no_grad ():
        output = model.forward(im)
        
    output_proba = torch.exp(output)
    
    probs, idx = output_proba.topk(topk)
    probs = np.array(probs)[0]
    idx = np.array(idx)[0]
    
    idx_to_class = {v:k for k,v in model.class_to_idx.items()}
    classes = np.array([idx_to_class[item] for item in idx])
    
    return probs, classes

### MAIN
if __name__ == "__main__":
     
    ### Choose gpu or cpu device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    ### Load classe names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    ### Display user parameters
    print('image_dir: {}\ncheckpoint_dir: {}\ntop_k: {}\ncategory_names: {}\ndevice: {}'.format(
        args.image_dir, args.checkpoint_dir+'.pth', args.top_k, args.category_names, device
    )) 
    
    ### Predict class
    
    # Load model and make prediction
    model = loading_model(args.checkpoint_dir+'.pth')
    img = process_image(args.image_dir)

    probs, classes = predict(args.image_dir, model, args.top_k, device)
    
    class_names = [cat_to_name[item] for item in classes]
    probs_str = ["{:.2f}% ".format(i*100) for i in probs]
    
    # Display into a pandas DataFrame
    print(
        pd.DataFrame({
        'Class Name':class_names,
        'Probability':probs_str
    }))
    