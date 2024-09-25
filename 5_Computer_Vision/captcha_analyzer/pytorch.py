import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import string

CAPTCHA_LEN=5
IMAGE_WIDTH=200
IMAGE_HEIGHT=50


"""
Custom Dataset Class. inherits from PyTorch's Dataset 

This class allows you to customize how data (images and labels) 
    is loaded, transformed, and accessed.
"""
class Captcha_Dataset(Dataset):
    def __init__(self, directory):
        self.directory=directory

        # find all files in "directory" that match with ".png" extension.
        self.image_paths=glob.glob(os.path.join(directory, '*.png'))                
        if len(self.image_paths)==0:
            raise ValueError(f"No images found in the directory: {directory}")
        else:
            print(f'Found {len(self.image_paths)} images in the directory.')

        # transformations are applied to the images in a dataset 
        #   to prepare them for use in a neural network.
        self.transform=transforms.Compose([      # Combines multiple transformation operations
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),        # Resizes the images to a specific size
            transforms.ToTensor(),               # Image to tensor
            transforms.Normalize((0.5,), (0.5,)) 
            # Normalizes by applying a mean (0.5) and standard deviation (0.5).
        ])

        # alphanumerical characters
        self.characters  =string.ascii_letters+string.digits        
        self.char_to_idx ={ # dic of index
            ch: idx for idx, ch in enumerate(self.characters)
        }
    
    """
    @Override
    Returns the total number of items in the dataset.
    """
    def __len__(self): return len(self.image_paths)
    
    """
    @Override    
    Get an image using an index.

    Args:
        idx (int) : Index of an image.
    """
    def __getitem__(self, idx):

        path  =self.image_paths[idx]        
        # open an image and convert it to grayscale mode
        image =Image.open(path).convert('L')
        # transform the image
        image =self.transform(image)
        
        # get the label from the file name
        label = os.path.basename(path).split('.')[0]  
        
        # ensure the filename matches with the number of characters in the captcha
        if len(label) != CAPTCHA_LEN:
            raise ValueError(f"Label must be of length 5, but got: {label}")
        
        # convert the label to an array of indices
        label_idx    =[self.char_to_idx[ch] for ch in label]          
        label_tensor =torch.tensor(label_idx) # array to tensor
        
        return image, label_tensor  




"""
Convolutional Neural Network. Image classification tasks

"""
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        
        """first Convolutional layer.
        PARAMETERS:
        1           : images with 1 channel (grayscale)
        32          : convolution filters
        kernel_size : each filter of size 3x3
        padding     : ensures the output dimensions of the image remains the same
        """
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        """second Convolutional layer."""
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        
        # This defines a max pooling layer that down-samples the 
        #   input by taking the maximum value in each 2x2 block of pixels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        """Fully Connected layers.
        64: output size of the previous layer
        12: height of the image after applying maxpool to 
            the 2 convolutional layers. ((50/2)/2)
        50: width of the image after applying maxpool to 
            the 2 convolutional layers. ((200/2)/2)
        """
        self.fc1 = nn.Linear(64*12*50, 256)          
        self.fc2 = nn.Linear(256, num_classes) 
        """
        num_classes = 62. number of different chars in the captcha
            the predicted char in each image is calculated by
            looking for the greatest value among the 62 outputs.
        """


    """
    Forward pass of the neural network.

    Args:
        x (Torch) : batch of grayscale images.
    """
    def forward(self, x):
        
        """convolution layers        
        [input_size, output_size, height, width]

        input image size is  [12, 1, 50, 200],
        output image size is [12, 32, 25, 100]        
        the dimensions are divided by half using the pooling layer.
        """        
        x=self.pool(torch.relu(self.conv1(x)))  
        x=self.pool(torch.relu(self.conv2(x)))  #[12, 64, 12, 50]

        # 3D tensor output from the convolutional layers is flattened 
        #   into a 2D tensor where each row corresponds to an image
        x=x.view(x.size(0), -1)  # Flatten
        
        """fully connected layers"""
        x=torch.relu(self.fc1(x))  # Expecting 38400 inputs
        x=self.fc2(x)


        return x



"""
Training function

Args:
    model (CNN)                  : Neural network.
    dataloader (DataLoader)      : Dataloader with all the images.
    criterion (CrossEntropyLoss) : Entropy loss criterion.
    optimizer (Adam)             : Optimizer. Backtracking function
    num_epochs (int)             : Number of epochs.
"""
def train(model, dataloader, criterion, optimizer, num_epochs):
    
    # iterates throw the images 'epochs' times
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            # resets the gradients of all model parameters to zero
            #   gradients accumulate by default in PyTorch
            optimizer.zero_grad()

            # forward 
            outputs=model(images)  # shape: [12, 62]
            
            
            """# check shapes
            print(f"\nBEFORE.\nShape: {outputs.shape}")  # Should be [12, 62]
            print(f"Labels: {labels.shape}")             # Should be [12, 5]
            """

            # unsqueeze() : Adds an extra dimension to the outputs
            # expand()    : Expands this tensor to repeat the single dimension
            # reshape()   : Flattens the first two dimensions into one. 5*12=60
            outputs_flat =outputs.unsqueeze(1).expand(-1, CAPTCHA_LEN, -1).reshape(-1, outputs.size(-1))  # Shape: [60, 62]
            # flattens the labels from [12, 5] to [60],
            labels_flat  =labels.view(-1)  # Shape: [60]

            
            """# check shapes
            print(f"\nAFTER.\nShape: {outputs_flat.shape}")  # Should be [60, 62]
            print(f"Labels: {labels_flat.shape}")            # Should be [60]
            """

            # calculate loss
            loss=criterion(outputs_flat, labels_flat)  
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")









"""
Evaluation function
"""
def evaluate():
    """TODO"""




def main():
    idx=0
    directory='images_0/'

    learning_rate=0.001
    num_epochs=100

    model       =CNN(num_classes=62)
    dataset     =Captcha_Dataset(directory)
    dataloader  =DataLoader(dataset, batch_size=32, shuffle=True)
    
    
    criterion =nn.CrossEntropyLoss()
    optimizer =optim.Adam(model.parameters(), lr=learning_rate)
    
    train(model, dataloader, criterion, optimizer, num_epochs=num_epochs)
    
    

if __name__ == '__main__':
    main()
