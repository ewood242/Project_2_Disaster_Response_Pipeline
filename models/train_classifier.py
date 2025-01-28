import sys
import pandas as pd
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
import xgboost as xgb
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Parameter:
    database_filepath (str): path to the SQLite database

    Returns:
    X (pandas.DataFrame): contains the feature variables
    Y (pandas.DataFrame): contains the target variables
    category_names (list): contains the category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("DisasterResponseMessages", con=engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = Y.columns.tolist()
    return X,Y,category_names


def tokenize(text):
    """
    Tokenize the text.

    Parameter:
    text (str): text to tokenize and lemmatize

    Return:
    tokens (list): list of tokens resulting from text processing
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return tokens


def build_model():
    """
    Build ML model and adjust parameters.

    Return:
    cv (GridSearchCV): object including model pipeline and grid of parameters
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(xgb.XGBClassifier()))
    ])
            
    parameters = {
        'clf__estimator__n_estimators' : [0, 200],
        'clf__estimator__max_depth': [2, 5, 10],
        'clf__estimator__subsample': [0.25, 0.5, 0.75, 1]
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance.

    Parameters:
    model: trained model
    X_test (pandas.DataFrame): feature variables for testing
    Y_test (pandas.DataFrame): target variables for testing
    category_names (list): contains the category names
    """
    y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, y_pred, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    """
    Save the trained model.

    Parameters:
    model: the trained model
    model_filepath (str): path to model save location
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Load and tokenize data, build and train model, evaluate and save model.
    """
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
        print('Please provide the filepath of the disaster response messages '\
              'database as the first argument and the filepath of the pickle '\
              'file to save the model to as the second argument.'\
              '\n\nExample: python train_classifier.py '\
              '../data/DisasterResponseMessages.db classifier.pkl')


if __name__ == '__main__':
    main()