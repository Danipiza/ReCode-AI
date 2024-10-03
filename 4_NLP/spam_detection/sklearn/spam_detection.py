# Step 1: Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report




def main():
    
    # -- load dataset ----------------------------------------------------------
    path='../data/SMSSpamCollection.tsv'
    data=pd.read_csv(path, 
                     sep='\t', header=None, 
                     names=['label', 'message'])


    # -- convert labels --------------------------------------------------------
    # (spam or ham) to binary values (spam = 1, ham = 0)
    data['label']=data['label'].map({'spam': 1, 'ham': 0})

    
    # -- dataset -> training and testing sets ----------------------------------
    X_train, X_test, \
    y_train, y_test=train_test_split(
        data['message'], data['label'], 
        test_size=0.2, random_state=42
    )

    
    # -- text vectorization (TF-IDF) -------------------------------------------
    tfidf=TfidfVectorizer(stop_words='english', max_df=0.9)
    
    X_train_tfidf=tfidf.fit_transform(X_train)
    X_test_tfidf =tfidf.transform(X_test)

    
    # -- train Naives Bayes classifier -----------------------------------------    
    model=MultinomialNB()
    model.fit(X_train_tfidf, y_train)
        
    # -- predictions -----------------------------------------------------------        
    y_pred=model.predict(X_test_tfidf)

            
    # -- evaluation ------------------------------------------------------------        
    # Step 8: Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    
    # -- new message evaluation ------------------------------------------------    
    new_message=["Congratulations! You've won a free ticket to Bahamas! Click here to claim."]
    new_message_tfidf=tfidf.transform(new_message)
    prediction=model.predict(new_message_tfidf)
    print("\nNew Message Prediction (1 = spam, 0 = not spam):", prediction[0])

main()
