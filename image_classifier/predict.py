

import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import set_device
from torchvision import models

from workspace_utils import active_session


def arg_parsing():
    """
    Parses arguments from command line
    """

    # parser instance
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # select image
    parser.add_argument('--image', 
                        type=str, 
                        help='Select testing image.',
                        required=True)

    # load checkpoint
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Checkpoint file path.',
                        required=True)
    
    # top-k classes
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Select top K matches.')
    
    # load category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping categories names.',
                        required=True)

    # set gpu option
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU.')

    # args
    args = parser.parse_args()
    
    return args


def load_checkpoint(checkpoint_path):
    """
    Loads deep learning model checkpoint.
    """
    
    checkpoint = torch.load(checkpoint_path)
    architecture = checkpoint['model']

    model_selected = getattr(models, architecture)
    model = model_selected(pretrained=True)
    model.name = architecture

    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    
    # load data from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    
    # reading PIL image
    input_image = PIL.Image.open(image_path)

    # input image dimensions
    w, h = input_image.size

    # resizing
    resize=[256**600, 256]
    if w < h: 
        resize=[256, 256**600]
        
    input_image.thumbnail(size=resize)

    # cropping
    image_center = w/4, h/4
    left = image_center[0]-(224/2)
    top = image_center[1]-(224/2)
    right = image_center[0]+(224/2)
    bottom = image_center[1]+(224/2)  
    cropped_image = input_image.crop((left, top, right, bottom))

    # to numpy
    np_image = np.array(cropped_image)/255 
    
    # normalize channels
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    normalise_image = (np_image-normalise_means)/normalise_std
        
    # set first channel as color
    final_image = normalise_image.transpose((2, 0, 1))
    
    return final_image



def predict(image_tensor, model, device, topk):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # deactivating gpu for prediction
    model.to(device)
    
    # setting model for evaluation
    model.eval()
    
    # type
    tensor_type = torch.cuda.FloatTensor if device else torch.FloatTensor

    # Convert image from numpy to torch
    image = torch.from_numpy(np.expand_dims(image_tensor, axis=0)).type(tensor_type)

    # finding probabilities
    fc_out = model(image)
    
    # to linear scale
    linear_fc_out = torch.exp(fc_out)

    # topk
    default_topk = 5
    topk = topk or default_topk

    # top 5 results
    probs, classes = linear_fc_out.topk(topk)
    
    # detaching detail
    probs = np.array(probs.detach())[0] 
    classes = np.array(classes.detach())[0]
    
    # converting to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    labels = [idx_to_class[c] for c in classes]
    
    return probs, labels


# Main function
def main():
    """
    Executing predictions
    """
    
    # command line args
    args = arg_parsing()
    
    # load categories
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # load trained model
    model = load_checkpoint(args.checkpoint)
    
    # process Image
    image_tensor = process_image(args.image)
    
    # set device
    device = set_device(gpu_arg=args.gpu)
    
    # prediction
    probs, labels = predict(image_tensor, model, device, args.top_k)
    
    # flowers names
    flowers = [cat_to_name[lab] for lab in labels]

    # print probabilities
    for i, j in enumerate(zip(flowers, probs)):
        print(
            "Rank {}:".format(i+1),
            "Flower: {}, likelihood: {}%".format(j[0], ceil(j[1]*100))
        )

# program excecution
if __name__ == '__main__': 
    print('making prediction..') 
    with active_session():
        main()
    print('prediction finished.')