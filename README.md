# Project_2_Disaster_Response_Pipeline
Udacity Data Scientist Nanodegree Project #2 of 4: Disaster Response Pipeline

Table of Contents:

1. Libraries
2. Project Motivation
3. Respository Files
4. Running the Code
5. Acknowledgements

Libraries & Installation

This project uses Python version 3.* and the following libraries:

* pandas
* nltk
* sys
* sklearn
* sqlalchemy
* pickle


Project Motivation

This project was the second of four in the Udacity Data Scientist Nanodegree Program. The purpose of this project is to develop a web app that leverages machine learning to rapidly classify messages sent during a disaster into a set of 36 pre-defined categories. Using data from Figure Eight, the goal was to develop ETL and ML pipelines to perform multi-label classification. 


Repository Files

app:
* master.html: template for the main page of the web app
* go.html: template for the classification result page of the web app
* run.py: Flask template file used to run the web app

data:
* disaster_messages.csv: messages data
* disaster_categories.csv: categories data
* DisasterResponseMessages.db: database to save cleaned and categorized data
* process_data.py: module used to clean and store the data into a database

models:
* train_classifier.py: module used to build, fit, evaluate, and save the machine learning model used for classification
* model.pkl: the model output from train_classifier.py

notebooks:
* categories.csv: categories data used for the ETL Pipeline Preparation notebook
* messages.csv: messages data used for the ETL Pipeline Preparation notebook
* ETL Pipeline Preparation.ipynb: notebook used to prep the process_data.py module
* ML Pipeline Preparation.ipynb: notebook used to prep the train_classifier.py module


Running the Code
Within the project's root directory:
* To clean and store the data in a database, run the ETL pipeline by copying and pasting the following command into the terminal: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseMessages.db
* To train the classifier and save the resulting model, run the ML pipeline by copying and pasting the following command into the terminal: python models/train_classifier.py data/DisasterResponseMessages.db models/model.pkl
* Run the following command in the app's directory to run the web app: python app/run.py


Acknowledgements
The data used in this project was provided by Figure Eight: https://appen.com/
