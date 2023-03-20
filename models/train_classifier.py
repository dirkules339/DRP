import sys
import pandas as pd
import os
import nltk
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.model_selection import train_test_split
import re
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import pickle

def load_data(database_filepath):
    database_filepath = "../data/disaster_response_db.db"
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(table_name,engine)
    
    df = df.drop('child_alone',axis=1)
    
    df['related'] = df['related'].replace(2,1)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns
    
    return X,y,category_names


def tokenize(text,url_place_holder="url"):
    """
    Function that tokenizes text input
    Arguments: text -> Text to tokenize
    Output:final_tokens -> List of extracted tokens
    """
    
    # Replace all urls with placeholder
    url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls 
    urls = re.findall(url, text)
    
    # Replace url with a url placeholder string
    for urls in urls:
        text = text.replace(urls, url_place_holder)

    # Extract the word tokens 
    tokens = nltk.word_tokenize(text)
    
    #Lemmatizer to remove not neccassary forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of tokens
    final_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return final_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    y_prediction_test = pipeline_fitted.predict(X_test)
    
    print(classification_report(y_test.values, y_prediction_test, target_names=y.columns.values))
    
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disaster_response_db.db classifier.pkl')


if __name__ == '__main__':
    main()