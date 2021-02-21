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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
nltk.download(['punkt', 'wordnet','stopwords'])


def tokenize(text):
    """ Follows the standard text processing flow """
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


class Model(object):

    def build_model(self):
        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

        return pipeline
