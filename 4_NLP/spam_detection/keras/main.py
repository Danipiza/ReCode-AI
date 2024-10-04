import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense
import pandas as pd
import numpy as np


"""
Preprocess text data: Tokenization and padding.

Args:    
    messages(string[]) : Divide list of train messages.
    max_len (int)      : Use for the padding.
"""
def preprocess_text(messages, max_len=100):
    tokenizer=Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(messages)
    
    sequences =tokenizer.texts_to_sequences(messages)
    padded    =pad_sequences(sequences, maxlen=max_len, 
                             padding='post', truncating='post')
    
    return padded, tokenizer

"""
Manual train-test split without using sklearn.

Args:
    data(string[])    : List with the phrases.
    labels(string[])  : List with the labels.
    test_size (float) : Percentage of the dataset reserved to the test.

Return:
    X_train(string[]) : Divide list of train messages.
    X_test(string[])  : Divide list of test messages.
    y_train(string[]) : Divide list of train labels.
    y_test(string[])  : Divide list of test labels.
""" 
def manual_train_test_split(data, labels, test_size=0.2):
    
    # -- shuffle the data ------------------------------------------------------
    indices=np.arange(len(data))
    np.random.shuffle(indices)
    
    # -- split index -----------------------------------------------------------
    split_index=int((1-test_size)*len(data))    
    
    # -- split the data and labels ---------------------------------------------
    X_train, X_test =data[indices[:split_index]], \
                     data[indices[split_index:]]
    y_train, y_test =labels[indices[:split_index]], \
                     labels[indices[split_index:]]
    
    
    return X_train, X_test, y_train, y_test



"""
Create the neural network model.

Args:
    vocab_size (int)    : Size of the vocabulary.
    embedding_dim (int) : Size of the word vectors that will be 
            created by the Embedding layer.
    max_len (int)       : Maximum length of the input sequences.

Return:
    model (keras model) : Spam detection Model.
"""
def create_model(vocab_size, embedding_dim=16, max_len=100):
    
    model=Sequential([
        # converts each word (represented by a unique integer index) 
        #   into a dense vector of fixed size (embedding_dim).
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        # takes the average of all word vectors in the input sequence, 
        #   resulting in a single vector for the entire message.
        GlobalAveragePooling1D(),
        # first fully connected layer with 24 neurons 
        #   and a ReLU activation function.
        Dense(24, activation='relu'),
        # second fully connected layer. output layer.
        Dense(1, activation='sigmoid')
    ])
    
    # binary cross-entropy loss function is used for 
    #   binary classification problem. (1: Spam, 0: Ham)
    model.compile(loss='binary_crossentropy', 
                  optimizer='adam', metrics=['accuracy'])
    
    return model

# Main function
def main():
    # -- load dataset ----------------------------------------------------------
    path='../data/SMSSpamCollection.tsv'
    dataset=pd.read_csv(path, 
                     sep='\t', header=None, 
                     names=['label', 'message'])
    
    # -- convert labels --------------------------------------------------------
    # Label encoding: ham -> 0, spam -> 1
    dataset['label'] = dataset['label'].map({'ham': 0, 'spam': 1})
    
    messages=dataset['message'].values
    labels=dataset['label'].values


    # -- dataset -> training and testing sets ----------------------------------        
    X_train, X_test, \
    y_train, y_test=manual_train_test_split(messages, labels, 
                                            test_size=0.2)
    # -- preprocess ------------------------------------------------------------        
    max_len=100
    train_padded, tokenizer=preprocess_text(X_train, max_len=max_len)
    test_padded=pad_sequences(tokenizer.texts_to_sequences(X_test), 
                              maxlen=max_len, 
                              padding='post', truncating='post')
    
    # -- create and train the model --------------------------------------------      
    vocab_size=10000
    model=create_model(vocab_size, embedding_dim=16, max_len=max_len)    
    model.fit(train_padded, y_train, epochs=10, 
              validation_data=(test_padded, y_test), verbose=2)
        
    
    # -- evaluation ------------------------------------------------------------    
    loss, accuracy = model.evaluate(test_padded, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
