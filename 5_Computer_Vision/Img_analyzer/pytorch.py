import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd 
import os

import time

"""
Dataset.

Args:
    csv_file (string)                 : Path to the csv file. 
    root_dir (string)                 : Path of the directory.
    tranform (torchvision.transforms) : Function applied in __getitem__.
"""
class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data     =pd.read_csv(csv_file)
        self.root_dir =root_dir

        self.transform = transform

        # -- Possible categories -------------------------------------------------------
        self.strings=['Trouser', 'Pullover', 'Sneaker', 'Shirt', 'Bag', 
                      'Ankle boot', 'T-shirt', 'Coat', 'Sandal', 'Dress']
        
        self.string_to_idx={
            'Trouser':0, 'Pullover':1, 'Sneaker':2, 
            'Shirt':3, 'Bag':4, 'Ankle boot':5, 'T-shirt':6, 
            'Coat':7, 'Sandal':8, 'Dress':9
        }        
        # ------------------------------------------------------------------------------

    """Get length"""
    def __len__(self): return len(self.data)

    """
    Getter.

    Args:
        idx (int) : Index.
    
    Return:
        image (Image)         : Image of the idx-th item in the dataset.
        label_tensor (tensor) : Label of the idx-th item in the dataset.
    """
    def __getitem__(self, idx):
        image_path=os.path.join(self.root_dir, self.data.iloc[idx, 1])
        
        image=Image.open(image_path).convert('L') # grayscale
        label=self.data.iloc[idx, 0]

        if self.transform: image=self.transform(image)        

        # convert the label to a categorical index.
        label_idx    =self.string_to_idx[label]
        label_tensor =torch.tensor(label_idx) # int to tensor
        
        return image, label_tensor

"""
Convolutional Neural Network. Image classification tasks.
"""
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        """first Convolutional layer.
        PARAMETERS:
        1           : images with 1 channel (grayscale)
        32          : convolution filters
        kernel_size : each filter of size 3x3
        padding     : ensures the output dimensions of the image remains the same
        """
        self.conv1=nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # This defines a max pooling layer that down-samples the 
        #   input by taking the maximum value in each 2x2 block of pixels
        self.pool =nn.MaxPool2d(kernel_size=2, stride=2)
        
        """Fully Connected layers.
        64: output size of the previous layer
        7: height of the image after applying maxpool to 
            the 2 convolutional layers. ((28/2)/2)
        7: width of the image after applying maxpool to 
            the 2 convolutional layers. ((28/2)/2)
        """
        self.fc1  =nn.Linear(64 * 7 * 7, 128)
        self.fc2  =nn.Linear(128, num_classes)        
        #num_classes = 10. number of different clothes.
        
        
        
    """
    Forward pass of the neural network.

    Args:
        x (Torch) : batch of grayscale images.
    """
    def forward(self, x):
        # convolution layers
        x =self.pool(torch.relu(self.conv1(x)))        
        x =self.pool(torch.relu(self.conv2(x)))

        # 3D tensor output from the convolutional layers is flattened 
        #   into a 2D tensor where each row corresponds to an image        
        x=x.view(x.size(0), -1)
        
        # fully connected layers
        x=torch.relu(self.fc1(x))        
        x=self.fc2(x)  
        
        return x

"""
Evaluation function.

Args: 
    model (CNN)             : Neural Network model.
    dataloader (DataLoader) : DataLoader with all the images and labels.
"""
def evaluate_model(model, dataloader):
    model.eval()  # model to evaluation mode

    correct=0
    total  =0

    # disable gradient calculation. dont need to calculate or store gradients,.
    # only required during training for backpropagation.
    with torch.no_grad():  
        
        for images, labels in dataloader:
            images=images.float()

            # forward
            outputs=model(images)            
            _, predicted = torch.max(outputs.data, 1)  # get the predicted class

            # count the correct predictions of the model. 
            # tensor comparation.             
            correct +=(predicted==labels).sum().item()  
            total   +=labels.size(0)  # batch_size
            
                

    return 100*correct/total



def main():

    time_start=time.time()

    # -- hyperparameters -----------------------------------------------------------
    batch_size    =50
    learning_rate =0.001
    num_epochs    =10
    
    
    # -- transforms applyied in _getitem_() function of the dataset ----------------
    
    transform=transforms.Compose([
        transforms.Resize((28, 28)),         # ensure dimension
        transforms.ToTensor(),               # tensor
        transforms.Normalize((0.5,), (0.5,)) # normalize
    ])

    
    # -- create dataset and dataloader ---------------------------------------------     
    csv_file ='data/data.csv'
    root_dir ='data/'      

    dataset    =ImageDataset(csv_file, root_dir, transform=transform)
    dataloader =DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    num_classes=len(dataset.strings)
    
    # -- CNN -----------------------------------------------------------------------        
    model     =SimpleCNN(num_classes=num_classes)
    criterion =nn.CrossEntropyLoss()
    optimizer =optim.Adam(model.parameters(), lr=learning_rate)
    
    
    time_end=time.time()
    print("Finish Init! \tTime needed: {}s".format(time_end-time_start))
    
    # -- Training loop -------------------------------------------------------------            
    time_start=time.time()

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):        
            # resets the gradients of all model parameters to zero
            #   gradients accumulate by default in PyTorch
            optimizer.zero_grad()
                   
            # forward 
            outputs=model(images)
                       
            loss=criterion(outputs, labels)
            
            # backward
            loss.backward()
            optimizer.step()

        # evaluate the model after an epoch
        accuracy=evaluate_model(model, dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] - Accuracy: {accuracy:.2f}%')
    time_end=time.time()

    print("Finish!!\nExecution time:", time_end-time_start)


main()