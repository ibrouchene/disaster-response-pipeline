# import libraries
import sys
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import pickle as pk
# Lets import the machine learning stuff
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

CURRENT_MODEL = r".\model_results\model_d"
if CURRENT_MODEL not in sys.path:
    sys.path.append(CURRENT_MODEL)

import model as m


def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message.values
    # We are only interested in the categories
    start_column_index = df.columns.get_loc("related")
    Y = df.iloc[:, start_column_index:].values
    return X, Y, list(df.columns.values)[start_column_index:]


def evaluate_model(mod, x_test, y_test, cat_names):
    # predict on test data
    y_pred = mod.predict(x_test)
    df = get_results(y_test, y_pred, cat_names)
    df.to_csv(os.path.join(CURRENT_MODEL, r".\model_results.csv"), sep=";")


def save_model(mod, model_filepath):
    pk.dump(mod, open(model_filepath, "wb"))


def get_results(y_test, y_pred, cat_names):
    res = sklearn.metrics.classification_report(y_test, y_pred, output_dict=True, zero_division=0, target_names=cat_names)
    df = pd.DataFrame(res)
    return df.T


if __name__ == '__main__':

    X, Y, category_names = load_data('sqlite:///data/CleanDataBase.db')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    md = m.Model()
    model = md.build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    # model = pk.load(open(r".\model_no_grid_search.pkl", "rb"))

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(r".\model.pkl"))
    save_model(model, os.path.join(CURRENT_MODEL, r".\model.pkl"))

    print('Trained model saved!')


