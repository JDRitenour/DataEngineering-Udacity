# import libraries
import re
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import multioutput
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV


def load_data():
    """
    Loads the data from the DisasterResponse sqlite database.

    Returns:
        X: the messages to be classified
        y: columns with the classification options
        labels: the columns headers of the classification options
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.iloc[:, 4:]
    labels = list(df.columns[4:])
    return X, y, labels


def tokenize(text):
    """tokenizes the disaster response messages for training"""
    # remove special characters prior to tokenization
    text_cleaned = re.sub('[^A-Za-z0-9]+', ' ', text.lower().strip())

    # tokenize
    tokens = word_tokenize(text_cleaned)

    # define lemmatizer
    lemmatizer = WordNetLemmatizer()

    # define stop words
    stop_words = stopwords.words('english')

    # define list for cleaned tokens
    clean_tokens = []

    # define for loop to clean tokens
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok)
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Builds the model."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', multioutput.MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def model_results(y_test, y_pred, labels):
    """
    Prints the results of the model

    Parameters:
        y_test: the actual classifications from the test set
        y_pred: the predicted classifications from the model
        labels: the column names for the classifications
    """
    for i in range(len(labels)):
        print('Category: {} '.format(labels[i]))
        print(classification_report(y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])))


def main():
    X, y, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=96)

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_results(y_test, y_pred, labels)

    pickle.dump(model, open('model.pkl', 'wb'))


main()