import tensorflow as tf
from tensorflow.keras import layers, models

import pandas as pd

from PIL import Image

import os
import time
import numpy as np



"""
Dataset.

Args:
    csv_file (string)       : Path to the csv file. 
    root_dir (string)       : Path of the directory.
    tranform (def function) : Function applied in __getitem__.
    batch_size (int)        : Number of individuals per iteration.
"""
class ImageDataset(tf.keras.utils.Sequence):
    def __init__(self, csv_file, root_dir, transform=None, batch_size=50):
        self.data       =pd.read_csv(csv_file)
        self.root_dir   =root_dir
        self.batch_size =batch_size

        self.transform=transform        
        
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
        x=self.data.iloc[idx]

        image_path=os.path.join(self.root_dir, x[1])        

        image=Image.open(image_path).convert('L')  # grayscale
        label=self.string_to_idx[x[0]]

        if self.transform: image=self.transform(image)

        image=np.expand_dims(image, axis=-1)  # add channel dimension for grayscale

        return image, label

"""
Transform function: 
- resize
- convert to a numpy array
- normalize

Args:
    image (Image) : An image.

Return:
    image (Image) : The image with the transformations.
"""
def transform(image):
    image=image.resize((28, 28))
    image=np.array(image)/255.0  # normalize
    
    return image

"""
Convolutional Neural Network. Image classification tasks.

Args: 
    num_classes (int) : Number of classes of the images.

Return:
    model (tensorflow.keras) : CNN model.
"""
def SimpleCNN(num_classes):
    model=models.Sequential([        
        # first Convolutional layer.
        #        filters, size of each filter, input_size=output_size, activation function
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',  
                      input_shape=(28, 28, 1)), # input data shape (width, height, color channel)
        layers.MaxPooling2D((2, 2)), # This layer reduces the spatial dimensions

        # second Convolutional layer.
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # reshapes the multi-dimensional output from the 
        # previous layers into a one-dimensional vector
        # width*height*(output_size=64 from the previoys layer)
        # 7 * 7 * 64 = 3136
        layers.Flatten(),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # use softmax for classification
        # the activation function converts the output from the 
        # previous layer into probabilities that sum to 1
    ])
    return model

"""
Preprocessing function. Transform image and label to tensors.

Args:
    image (np array) : Image of an item.
    label (np array) : Label of an item.
Return:
    image (tensor)   : The image converted into a tensor
    label (tensor)   : The label converted into a tensor
"""
def preprocess(image, label):
    image=tf.convert_to_tensor(image, dtype=tf.float32)
    label=tf.convert_to_tensor(label, dtype=tf.int32)

    return image, label

"""
Function to evaluate the model

Args:
    model (tensorflow.keras) : CNN model.
    dataset (DataSet)        : Class with all the images.

Return:
    ret (float) : Accuracy of the model
"""
def evaluate_model(model, dataset):
    correct=0
    total  =0
    
    
    for idx in range(len(dataset)):
        image, label=dataset[idx]

        image=np.expand_dims(image, axis=0)  # add batch dimension

        prediction      =model.predict(image)
        predicted_class =np.argmax(prediction, axis=-1)
        
        correct+=int(predicted_class==label)
        total  +=1

    ret=correct/total*100
    return ret

def main():
    idx=1

    time_start=time.time()

    # -- hyperparameters -----------------------------------------------------------
    batch_size    =50
    learning_rate =0.001
    num_epochs    =10

    # -- create dataset and dataloader ---------------------------------------------
    csv_file ='data/data.csv'
    root_dir ='data/'

    dataset=ImageDataset(csv_file, root_dir, transform=transform)

    # tensorflow dataset 
    train_dataset=tf.data.Dataset.from_generator(
        lambda: (dataset[i] for i in range(len(dataset))),
        output_signature=(
            # shape of the images
            tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32), 
            # shape of the label (classes)
            tf.TensorSpec(shape=(), dtype=tf.int32) 
        )
    # preprocess the items of the dataset (convert into tensors)
    ).map(preprocess).batch(batch_size)


    num_classes=len(dataset.strings)

    # -- CNN -----------------------------------------------------------------------        
    model=SimpleCNN(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # -- Train model -----------------------------------------------------------    
    model.fit(train_dataset, epochs=num_epochs)

    # -- Evaluate model --------------------------------------------------------    
    accuracy=evaluate_model(model, dataset)
    print(f'Accuracy on the dataset: {accuracy:.2f}%')

    model.save('models/model_{}.h5'.format(idx))
    

if __name__ == "__main__":
    main()
