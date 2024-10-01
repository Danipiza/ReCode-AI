import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model


"""
Chatbot, processes user input, and predicts responses.

Args:
    intents_file (string) : Path to the intents files.
    words_file (string)   : Path to the words file.
    classes_file (string) : Path to the classes file.
    model_file (string)   : Path to the keras model.
"""
class ChatBot():

    def __init__(self, intents_file, words_file, classes_file, model_file):
                        
        self.lemmatizer=WordNetLemmatizer()

        # files generated in model.py
        self.intents =json.loads(open(intents_file).read())
        self.words   =pickle.load(open(words_file, 'rb'))
        self.classes =pickle.load(open(classes_file, 'rb'))
        self.model   =load_model(model_file, compile=False)

    """
    Main method for the chatbot. 
    It takes the user's message, 
    

    Args:
        message (string) : Message writen on the terminal.
    """
    def response(self,message):
        ints =self.predict_class(message) # predict the class
        ret  =self.get_response(ints)     # get the chatbot's response 
        
        return ret
    
    """
    Matches the predicted intent class (tag) with the responses stored in 
    the intents.json file and selects a random response from that intent.

    Args:
        tag (string) : Predicted tag.
    
    Return:
        ret (string) : Response of the chatbot.
    """
    def get_response(self, tag):        
        ret=""
        
        for i in self.intents['intents']:
            if i["tag"]==tag:
                ret=random.choice(i['responses'])
                break

        return ret    
    
    """
    Takes the sentence and predicts the intent. (FORWARD function)
    Uses the bag of words array as input to the trained Keras model.

    Arg;
        sentence () : Given sentence.

    Return:
        category (string) : predicted intent class (tag).
    """
    def predict_class(self, sentence):
        bow =self.bag_of_words(sentence)
        ret =self.model.predict(np.array([bow]))[0]

        max_index =np.where(ret==np.max(ret))[0][0]
        category  =self.classes[max_index]

        return category

    
    """
    Converts a sentence into a "bag of words" array.
    
    Each word from the sentence is mapped to a 1 or 0 
    depending on its presence in the pre-trained words list.

    Args:
        sentence (string) : Given sentence

    Return:
        bag (numpy int array) : Mapped bag of words.
    """
    def bag_of_words(self, sentence):
        sentence_words =self.clean_sentence(sentence)
        bag            =[0 for _ in range(len(self.words))]

        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word==w: bag[i]=1

        #print(bag)
        return np.array(bag)
    

    """
    This method tokenizes a sentence into words and lemmatizes them. 
    Lemmatizing converts words into their base form.
        - Example, "running" becomes "run".

    Args:
        sentence (string) : Sentence to be cleaned.

    Return:
        ret (string[]) : List of cleaned words in the sentence.
    """
    def clean_sentence(self, sentence):        
        ret =nltk.word_tokenize(sentence)
        ret =[self.lemmatizer.lemmatize(word) for word in ret]

        return ret
    
    

    

    
    

    

def main():
    chat_bot=ChatBot('intents.json','words.pkl',
                     'classes.pkl','chatbot_model.h5')

    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input=input()
        if user_input.lower()=='quit':
            print("Chatbot: Goodbye!")
            break

        print(chat_bot.response(user_input))

main()