# disaster-response-pipeline
Repository for the Disaster Response Pipeline project of the Data Scientist Udacity Nanodegree program

## Prerequisites
sklearn >= 0.20 is required to run the scripts

## Project overview
Given a data set consisting of text messages and 36 labeled features, we build a data pipeline that extracts, transforms and loads the data into an SQLite database file. We then let a machine learning pipeline run, taking the contents of the SQLite database as an input and producing a trained model stored in the form of a python pickle file. Last but not least, the web app can be used to try and classify custom messages.

## How to run the scripts
### How to run the ETL pipeline
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
### How to run the ML pipeline
 `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
### How to run the web app
 `python run.py`
