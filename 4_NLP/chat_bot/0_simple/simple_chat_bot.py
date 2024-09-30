import nltk
from nltk.chat.util import Chat, reflections



def main():
    # "intents"
    pairs=[
        [r"hi|hello|hey", ["Hello!", "Hi there!", "Hey!"]],
        [r"how are you?", ["I'm doing good, how about you?"]],
        [r"what is your name?", ["I am a chatbot created by you.", "My name doesn't matter, I'm here to help you!"]],
        [r"quit", ["Goodbye!"]],
        [r"thank you", ["You're welcome!", "Glad to help!"]],
    ]

    # create a chatbot instance
    chatbot=Chat(pairs, reflections)

    print("Hi! I'm a simple chatbot. Type 'quit' to exit.")
    chatbot.converse()

if __name__ == "__main__":
    main()
