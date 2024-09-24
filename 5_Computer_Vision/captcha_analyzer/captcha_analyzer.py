import os
import numpy as np
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.layers import BatchNormalization # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore


"""ONE-HOT
Encoding labels to one-hot format is a process commonly used
in machine learning and data classification.

Especially when working with multi-class classification problems. 
Here's a breakdown of its purpose and operation:

Purpose of one-hot encoding:
- Proper representation
- Implicit order removal
- Facilitates learning

How one-hot encoding works. Example: 
3 classes: "dog", "cat", "bird".

One-hot:
- "dog" becomes [1, 0, 0]
- "cat" becomes [0, 1, 0]
- "bird" becomes [0, 0, 1]
"""


class Neural_Network():

    def __init__(self):

        # Parameters

        self.IMG_WIDTH       =200 
        self.IMG_HEIGHT      =50  
        self.MAX_CAPTCHA_LEN =5   # Maximum number of characters in the captcha
        self.input_shape     =(self.IMG_WIDTH, self.IMG_HEIGHT, 1) # Gray scale = 1
        

        self.CHAR_LIST   ="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        self.NUM_CLASSES =len(self.CHAR_LIST)  # Número total de caracteres posibles


        self.model=None
        self.create_model()
        

    """
    Create the sequential model of the neural network.            
    """
    def create_model(self):
        
        # Keras Sequential API
        self.model=Sequential()
        
        # First convolutional layer
        #  
        # 32 grids 3x3 of neurons
        self.model.add(Conv2D(32, (3, 3), input_shape=self.input_shape, activation='relu'))
        # Normalize the outputs of the previous layer
        self.model.add(BatchNormalization())
        # Reduces the dimensionality of the outputs of the convolutional layer.
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second convolutional layer
        #
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third convolutional layer
        #
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # Converts 2D features extracted by convolutional layers
        #   into a one-dimensional vector
        self.model.add(Flatten())
        
        # Fully connected layer with 512 neurons
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5)) # prevent overfitting
        
        # Output Layer. (1 for each character in the CAPTCHA)
        self.model.add(Dense(self.MAX_CAPTCHA_LEN*self.NUM_CLASSES, activation='softmax'))
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # OPTIONAL
        # return model
    
        

    """
    Load the images and labels from a directory. 
    The labels are the files names.
    
    Args:
        directory (string): Path of the directory     

    Return:
        images (Images[]): Array of Images
        labels (string[]): Array of labels
    """
    def load_data(self, directory):
        
        images=[]
        labels=[]

        # All the files from the directory
        for file in os.listdir(directory):
            if file.endswith('.jpg') or file.endswith('.png'):
                # -- ADD IMAGE -----------------------------------------------------------------
                image_path=os.path.join(directory, file)

                # Convert to grayscale
                image=Image.open(image_path).convert('L')  
                # Ensure the dimensions
                image=image.resize((self.IMG_WIDTH, self.IMG_HEIGHT))
                # Normalize
                image=np.array(image)/255.0  

                images.append(image)
                
                
                # -- ADD LABEL -----------------------------------------------------------------
                # Without the extension (.jpg or .png)
                label=file.split('.')[0]
                labels.append(label)
        
        return np.array(images), np.array(labels)


    """
    Encodes labels to one-hot format.
    
    Args:
        labels (string[]): Array of labels

    Return:
        ret (int[][]): Array of individuals
    """
    def encode_labels(self, labels):
        
        dic_char_num={char: i for i, char in enumerate(self.CHAR_LIST)}
        
        n=len(labels)

        # Init all the individuals with 0s.
        #   individual is a matrix with 5 rows and 62 characters (alphanumerical)
        ret = np.zeros((n, self.MAX_CAPTCHA_LEN, len(self.CHAR_LIST)), dtype=np.float32)
        
        # For each individual modify the index of the label
        for i, label in enumerate(labels):
            for j, c in enumerate(label):
                ret[i, j, dic_char_num[c]]=1.0

        # converts the array to a new form in which 
        #   each CAPTCHA image is represented as a single row.
        ret=ret.reshape(n, self.MAX_CAPTCHA_LEN*self.NUM_CLASSES)

        return ret
    
    """
    Decode the label from one-hot to string.
    
    Args:
        prediction (int[]): Array representing a prediction
    
    Return:
        :return: Texto del captcha predicho.
    """
    def decode_label(self, prediction):
        
        dic_char_num={i: char for i, char in enumerate(self.CHAR_LIST)}
        
        prediction=prediction.reshape(self.MAX_CAPTCHA_LEN, len(self.CHAR_LIST))  # Reformar la predicción para cada carácter
        ret=""

        for captcha_char in prediction:
            max_indx=0
            for i in range(len(captcha_char)):
                if captcha_char[i]==1: 
                    max_indx=i
                    break

            ret+=dic_char_num[max_indx]

        return ret

    """
    Train the model.

    Args:
        images (Images[])        : Array of Images.
        labels (string[])        : Array of labels.
        epochs (int)             : Number of epochs.
        batch_size (int)         : Batch size.
        validation_split (float) : Percentage of training data to be reserved 
                                    for validation during training
    """
    def train_model(self, images, labels,
                    epochs=100, batch_size=32, 
                    validation_split=0.1):
        
        self.model.fit(images, labels, 
                       epochs=epochs, batch_size=batch_size, 
                       validation_split=validation_split)

    """
    Evaluation of the model.
    
    Args:
        directory (string): Path to the directory with the evaluation images
    """
    def evaluation(self, directory):
                
        images, labels =self.load_data(directory)                
        images=np.expand_dims(np.array(images), axis=-1)
        
        predictions=self.model.predict(images)

        n=len(images)
        aciertos=0
        for i in range(n):
            decode_pred=self.decode_label(predictions[i])

            print(f"Real: {labels[i]} | Predicted: {decode_pred}")            
            if decode_pred==labels[i]: aciertos+=1
        
        
        precision=(aciertos/n)*100
        print(f"\nModel precision in {n} images: {precision:.2f}%")


def main():
    idx=1
    directory='images'
    nn=Neural_Network()    

   
    
    images, labels = nn.load_data(directory)

    # Add color channel for grayscale 
    images=np.expand_dims(images, axis=-1)  
    labels=nn.encode_labels(labels) # one-hot

    
   
    nn.train_model(images, labels, epochs=5)
    nn.model.save('models/keras/captcha_{}.keras'.format(idx))

    print("Model trained and saved successfully.\n")
    
    # evaluation
    ditectory='images_0'  
    nn.evaluation(ditectory)



main()