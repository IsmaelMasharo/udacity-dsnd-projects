# importing libraries
import argparse
from os.path import isdir

from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from workspace_utils import active_session

# arg parsing function
def arg_parsing():
    """
    Parses arguments from command line
    """

    # parser instance
    parser = argparse.ArgumentParser(description="Neural Network Settings")

    # select data directory
    parser.add_argument('--data_dir', 
                        type=str, 
                        help='Data directory.')
    
    # select architecture
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models')
    
    # select directory
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Saving directory path for checkpoints.')
    
    # set hyperparameters
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Gradient descent learning rate')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Number of hidden layers for DNN.')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training.')

    # set gpu option
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU.')
    
    # args
    args = parser.parse_args()
    
    return args



def transformation_for_training(train_dir):
    """
    Performs training transformations on dataset.
    """

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    return datasets.ImageFolder(train_dir, transform=train_transforms)
    

def transformation_for_testing(test_dir):
    """
    Performs test transformations on a dataset.
    """
    
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return test_data

def data_loader(data, train=True):
    """
    Creates a dataloader from dataset    
    """

    loader_config = {
        'batch_size':64,
        'shuffle':train
    }
    
    return torch.utils.data.DataLoader(data, **loader_config)


def set_device(gpu_arg):
    """
    Sets gpu or cpu as device.    
    """

    dev = 'cpu'
    if gpu_arg and torch.cuda.is_available():
        dev = 'cuda'
    elif gpu_arg:
        print('Not gpu found. Using cpu instead.') 

    return torch.device(dev)


def set_model_architecture(architecture=None):
    """
    Sets model architecture.
    """
    
    default_arch = "vgg16"
    selected_arch = architecture if architecture else default_arch

    model_selected = getattr(models, selected_arch)
    model = model_selected(pretrained=True)
    model.name = selected_arch

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    
    return model


def create_classifier(model, hidden_units=None):
    """
    Returns classifier with the specified hidden units.
    """

    defaul_nb_units = 4096
    nb_units = hidden_units if hidden_units else defaul_nb_units
    
    input_features = model.classifier[0].in_features
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, nb_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(nb_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    return classifier


def validation(model, testloader, criterion, device):
    """
    Validates model performance against testing set
    """
    test_loss = 0
    accuracy = 0
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        log_ps = model(inputs)
        test_loss += criterion(log_ps, labels)

        ps = torch.exp(log_ps)
        _, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy


def train_network(
        model, trainloader, testloader, device, 
        criterion, optimizer, epochs, print_every, 
        steps
    ):

    """
    Train neural network.
    """

    default_nb_epochs = 3 
    nb_epochs = epochs or default_nb_epochs
    
    # Train Model
    for e in range(nb_epochs):
        running_loss = 0
        model.train()
        
        for inputs, labels in trainloader:
            steps += 1        
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # eval mode for inference
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, testloader, criterion, device)
                
                print("Epoch: {}/{}.. ".format(e+1, nb_epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Validation Loss: {:.3f}.. ".format(valid_loss/len(testloader)),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                model.train()

    return model


def validate_model(model, testloader, device):
    """
    Final model validation
    """
    
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()

    print('Accuracy on test images: %d %%' % (100 * correct / total))
        

def create_checkpoint(model, save_dir, train_data):
    """
    Saves trained model to a specified save directory with the classifications
    of the training data.    
    """
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'model': model.name,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    if save_dir and isdir(save_dir):
        torch.save(checkpoint, save_dir + 'checkpoint.pth')
        print('checkpoint created')
    else: 
        print("Directory not found. Saving at current directory in checkpoint.pth")
        torch.save(checkpoint, 'checkpoint.pth')


# MAIN FUNCTION

def main():
    """
    Executing training.
    """
    
    # args
    args = arg_parsing()
    
    # directories
    data_dir = args.data_dir or 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # seting transforms and loaders
    train_data = transformation_for_training(train_dir)
    valid_data = transformation_for_training(valid_dir)
    test_data = transformation_for_testing(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # load model
    model = set_model_architecture(architecture=args.arch)
    
    # build classifier
    model.classifier = create_classifier(model, hidden_units=args.hidden_units)
     
    # check for GPU
    device = set_device(gpu_arg=args.gpu)
    
    # setting device
    model.to(device)
    
    # setting learning rate
    default_lr = 0.001
    learning_rate = args.learning_rate or default_lr

    # setting loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # seting params
    print_every = 50
    steps = 0
    
    # training
    trained_model = train_network(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    # model validation
    validate_model(trained_model, testloader, device)
    
    # saving checkpoint
    create_checkpoint(trained_model, args.save_dir, train_data)


# program excecution
if __name__ == '__main__':
    print('training started') 
    with active_session():
        main()
    print('training finished')
