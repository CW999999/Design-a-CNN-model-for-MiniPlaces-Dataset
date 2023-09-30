# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms
import math


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()
        # certain definitions
        w = input_shape[0]
        h = input_shape[1]

        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, bias=True)
        self.max_pool_1=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w = math.floor((w - 5 + 2) / 2)
        h = math.floor((h - 5 + 2) / 2)


        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=True)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        w = math.floor((w - 5 + 2) / 2)
        h = math.floor((h - 5 + 2) / 2)


        self.fc1 = torch.nn.Linear(16 * w * h,  256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 100)

    def forward(self, x):
        shape_dict = {}
        # certain operations
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.max_pool_1(x)
        shape_dict[1] = list(x.size())

        x = torch.nn.functional.relu(self.conv2(x))
        x = self.max_pool_2(x)
        shape_dict[2] = list(x.size())

        x = x.view(-1, 16*5*5)
        shape_dict[3] = list(x.size())

        x = torch.nn.functional.relu(self.fc1(x))
        shape_dict[4] = list(x.size())
        x = torch.nn.functional.relu(self.fc2(x))
        shape_dict[5] = list(x.size())
        out = self.fc3(x)
        shape_dict[6] = list(out.size())

        return out, shape_dict


def count_model_params():
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0
    for name, params in count_model_params():
        if params.requires_grad:
            model_params = model_params + params.numel()
    model_params = model_params/1000000


    return model_params



def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        ###################################
        # fill in the standard training loop of forward pass,
        # backward pass, loss computation and optimizer step
        ###################################

        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        # Uncomment the below line after you fill block 1 and 2
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
