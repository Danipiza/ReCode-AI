import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

class ChatBotModel():

    def __init__(self):
        self.lemmatizer=WordNetLemmatizer()

        self.intents=json.loads(open('intents.json').read())



        self.words          =[]
        self.classes        =[]
        self.documents      =[]
        self.ignore_letters =['?', '!', '¿', '.', ',']

        
        # classify patrons and categories
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, intent["tag"]))
                if intent["tag"] not in self.classes:
                    self.classes.append(intent["tag"])

        self.words =[self.lemmatizer.lemmatize(word) 
                      for word in self.words if word not in self.ignore_letters]
        self.words =sorted(set(self.words))

        pickle.dump(self.words, open('words.pkl', 'wb'))
        pickle.dump(self.classes, open('classes.pkl', 'wb'))

        self.trainning()
    
    def trainning(self):

        
        # information 0 and 1s present words in each category
        training=[]
        output_empty=[0 for _ in range(self.classes)]
        for document in self.documents:
            bag =[]

            word_patterns =document[0]
            word_patterns =[self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]

            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(document[1])]=1

            training.append([bag, output_row])

        random.shuffle(training)

        
        self.train_x=[]
        self.train_y=[]

        for i in training:
            self.train_x.append(i[0])
            self.train_y.append(i[1])

        self.train_x =np.array(self.train_x) 
        self.train_y =np.array(self.train_y)

    def create_model(self):

        # -- Neural Network ------------------------------------------------------------
        model=Sequential()

        model.add(Dense(128, input_shape=(len(self.train_x[0]),), name="inp_layer", activation='relu'))
        model.add(Dropout(0.5, name="hidden_layer1"))

        model.add(Dense(64, name="hidden_layer2", activation='relu'))
        model.add(Dropout(0.5, name="hidden_layer3"))
        
        model.add(Dense(len(self.train_y[0]), name="output_layer", activation='softmax'))


        sgd=SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])


        model.fit(np.array(self.train_x), np.array(self.train_y), epochs=100, batch_size=5, verbose=1)
        model.save("chatbot_model.h5")
