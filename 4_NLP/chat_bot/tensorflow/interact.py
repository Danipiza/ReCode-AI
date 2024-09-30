import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# load NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# load intents
with open('intents.json') as file:
    intents=json.load(file)

# load the trained model
model=load_model('chatbot_model.h5')

lemmatizer=WordNetLemmatizer()

# load the words from the saved file
with open('words.json') as file:
    words=json.load(file)

classes=[]

# prepare classes
for intent in intents['intents']:
    for pattern in intent['patterns']:
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# remove duplicates and sort
words=sorted(set([lemmatizer.lemmatize(w.lower()) for w in words]))

# debugging: print the number of words
print(f"Vocabulary size: {len(words)}")  # Print the size of the vocabulary

# function to preprocess input
def preprocess_input(user_input):
    bag=[0 for _ in range(len(words))]

    user_words =nltk.word_tokenize(user_input)
    user_words =[lemmatizer.lemmatize(w.lower()) for w in user_words]

    for w in user_words:
        if w in words: bag[words.index(w)]=1

    return np.array(bag)


def main():
    print("Chatbot is running! Type 'quit' to exit.")

    while True:
        user_input=input("You: ")
        if user_input.lower()=='quit':
            print("Chatbot: Goodbye!")
            break

        # preprocess the input and make prediction
        input_data=preprocess_input(user_input)

        # debugging: print the input data shape
        print(f"Input data shape: {input_data.shape}")  # Print shape for debugging

        prediction   =model.predict(np.array([input_data]))[0]
        intent_index =np.argmax(prediction)
        intent_tag   =classes[intent_index]

        # find the corresponding response
        for intent in intents['intents']:
            if intent['tag']==intent_tag:
                response = np.random.choice(intent['responses'])
                break

        print(f"Chatbot: {response}")

if __name__ == "__main__":
    main()
