# disaster-response-pipeline
Repository for the Disaster Response Pipeline project of the Data Scientist Udacity Nanodegree program

#### Project overview
Given a data set consisting of text messages and 36 labeled features (provided by [Figure 8](https://appen.com/)), we build a data pipeline that extracts, transforms and loads the data into an SQLite database file. We then let a machine learning pipeline run, taking the contents of the SQLite database as an input and producing a trained model stored in the form of a python pickle file. Finally, a web app can be used to classify messages.

# Getting started
sklearn >= 0.20 is required to run the scripts


# How to run the scripts
#### How to run the ETL pipeline
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
#### How to run the ML pipeline
 `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
#### How to run the web app
 `python run.py`
 
 # The data processing pipeline
 The data processing pipeline is a classical ETL pipeline and is implemented in the ***data/process_data.py*** file. Given a set of two csv files as an input (data/disaster_messages.csv and data/disaster_categories.csv) and a target database, the pipeline will:
  * Load the data from each csv file
  * Merge both dataframes into a single one
  * Reformat the categories such as to have each category on its own column
  * Binarize the categories' values
  * Remove duplicates & unexpected values
  * Save the final result in an SQLite database specified as an input argument

# The machine learning pipeline
The ML pipeline implements usual methods used in Natural Language Processing and is implemented in the ***model/train_classifier.py*** file. Given a database as an input and a target file location for a pickle file, the pipeline:
* Loads input data from the database mentioned in the input arguments
* Splits the dataset in a training set and a testing set
* Builds an ML model using a ML pipeline and using GridSearchCV for parameter selection
* Evaluates the model on the test dataset and generates a csv file with the results
* Saves the model in a pickle file

# The web app
The web app implementation is to be found in the ***app*** folder of the repository. Given an input message, the web app will load the trained model from the previously mentionned pickle file and output the results for each of the 36 categories. Once the web app is up and running, the app can be accessed by simply connecting to the 3001 port of your machine's local network IP adress (example: 192.168.1.2:3001)

# Comparing models
Getting the best possible classifier was not part of the scope of the project, however it is still a good idea to be able to compare more easily some models. For this purpose a folder with a specific structure has been added. For each model, it contains the pickle file, a text description of the model and the evaluation results of the model based on the test data set.
model_results
  - model_x
     - description.txt
     - classifier.pkl
     - model_results.csv  

For an easy manipulation of the data and easy visualization, the Jupiter notebook ***model_comparison.ipynb*** can be loaded. It contains methods that will loop through the model_results folder, read the model's results, format them and compares them.
