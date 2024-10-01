import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')


class ChatBot():

    def __init__(self):
        # load intents
        with open('intents.json') as file:
            self.intents=json.load(file)

        # load the trained model
        self.model=load_model('chatbot_model.h5')

        self.lemmatizer=WordNetLemmatizer()

        # load the words from the saved file
        with open('words.json') as file:
            self.words=json.load(file)

        self.classes=[]

        # prepare classes
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # remove duplicates and sort
        self.words=sorted(set([self.lemmatizer.lemmatize(w.lower()) for w in self.words]))

        # debugging: print the number of words
        print(f"Vocabulary size: {len(self.words)}")  # Print the size of the vocabulary

    # function to preprocess input
    def preprocess_input(self, user_input):
        bag=[0 for _ in range(len(self.words))]

        user_words =nltk.word_tokenize(user_input)
        user_words =[self.lemmatizer.lemmatize(w.lower()) for w in user_words]

        for w in user_words:
            if w in self.words: bag[self.words.index(w)]=1

        return np.array(bag)

    def execute(self):
        while True:
            user_input=input("You: ")
            if user_input.lower()=='quit':
                print("Chatbot: Goodbye!")
                break

            # preprocess the input and make prediction
            input_data=self.preprocess_input(user_input)

            # debugging: print the input data shape
            print(f"Input data shape: {input_data.shape}")  # Print shape for debugging

            prediction   =self.model.predict(np.array([input_data]))[0]
            intent_index =np.argmax(prediction)
            intent_tag   =self.classes[intent_index]

            # find the corresponding response
            for intent in self.intents['intents']:
                if intent['tag']==intent_tag:
                    response = np.random.choice(intent['responses'])
                    break

            print(f"Chatbot: {response}")


def main():
    print("Chatbot is running! Type 'quit' to exit.")
    chatbot=ChatBot()
    chatbot.execute()

    

if __name__ == "__main__":
    main()
