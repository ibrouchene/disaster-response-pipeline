# import libraries
import pickle as pk
import re
import sys

import nltk
import pandas as pd
# Lets import the machine learning stuff
import sklearn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    """ Loads the input data from the specified SQLite database file """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    # We are only interested in the categories
    start_column_index = df.columns.get_loc("related")
    Y = df.iloc[:, start_column_index:].values
    return X, Y, list(df.columns.values)[start_column_index:]


def tokenize(text):
    """ Follows the standard text processing flow
     1. Normalize: remove punctuation, set all to lowercase
     2. Tokenize
     3. Remove  stopwords (English dictionary expected)
     4. Stem & Lemmatize """
    # Normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]

    # Stem & Lemmatize
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in stemmed]

    return clean_tokens


def build_model():
    """ Setting up of the machine learning pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__max_df': (0.5, 1.0),
        'vect__max_features': (None, 5000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50,  200],
        'clf__estimator__min_samples_split': [2, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluates the model on the test data set and writes out a csv file with the results """
    y_pred = model.predict(X_test)
    df = get_results(Y_test, y_pred, category_names)
    df.to_csv(r".\model_results.csv", sep=";")


def save_model(mod, model_filepath):
    """ Saves the built model in a pickle file under the specified file path """
    pk.dump(mod, open(model_filepath, "wb"))


def get_results(y_test, y_pred, cat_names):
    """ Returns a dataframe containing the results out of the classification_report method of sklearn.metrics """
    res = sklearn.metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=cat_names)
    df = pd.DataFrame(res)
    return df.T


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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()