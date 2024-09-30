import json
import numpy as np

# Natural Language Toolkit (NLTK), a library 
#   for working with human language data.
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# lemmatization and model
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD



def main():
    
    # load intents (JSON file)
    with open('intents.json') as file:
        intents=json.load(file)

    lemmatizer=WordNetLemmatizer()
    words        =[] # store unique words from the patterns in the intents.
    classes      =[] # different tags.
    documents    =[] # tuples (tokenized word, corresponding tag).
    ignore_chars =['?', '!', '.', ','] # ignore characters

    
    # -- data preparation ------------------------------------------------------    
    for intent in intents['intents']: # through each intent
        for pattern in intent['patterns']: # pattern associated with the intent.
            word_list=nltk.word_tokenize(pattern)
            words.extend(word_list)

            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes: classes.append(intent['tag'])

    # post-processing
    words   =[lemmatizer.lemmatize(w.lower()) 
              for w in words if w not in ignore_chars]
    words   =sorted(set(words))
    classes =sorted(set(classes))

    
    # -- training data ------------------------------------------------------    
    
    training=[]
    output_empty=[0 for _ in range(len(classes))]

    # iterates through the documents
    for doc in documents:
        bag=[]
        word_patterns=doc[0]
        word_patterns=[lemmatizer.lemmatize(w.lower()) 
                       for w in word_patterns]
        
        # binary representation 
        for w in words:
            bag.append(1) if w in word_patterns else bag.append(0)

        # one-hot encoded output row where the index 
        #   corresponding to the intent is set to 1.
        output_row=list(output_empty)
        output_row[classes.index(doc[1])]=1

        training.append([bag, output_row])

    
    print('Training data length: {}'.format(len(training)))

    # content (debugging)
    for i, entry in enumerate(training):
        print('Entry {}: Bag: {}, Output row: {}'.format(i,entry[0],entry[1]))

    # efficient numerical operations using numpy.
    training=np.array(training, dtype=object)

    # creating input/output arrays
    try:
        train_x=np.array(list(training[:, 0]))
        train_y=np.array(list(training[:, 1]))
    except Exception as e:
        print("Error while creating train_x and train_y:", e)

        
    # -- model creation -----------------------------------------------------        
    if 'train_x' in locals() and 'train_y' in locals():
        
        model=Sequential() # init model
        
        # first layer. 128 units 
        #   (input shape correspond to the length of inputs train_x)
        model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        # prevent overfitting by randomly dropping some neurons during
        model.add(Dropout(0.5)) 
        
        # second layer. 64 units (half droped)
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        
        # output layer
        model.add(Dense(len(train_y[0]), activation='softmax'))

        # compile model with Stochastic Gradient Descent
        sgd=SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
        model.save('chatbot_model.h5')

        with open('words.json', 'w') as file:
            json.dump(words, file)

        print("Model training completed!")
    else:
        print("train_x or train_y could not be created.")


main()