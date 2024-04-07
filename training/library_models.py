# random forest imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# SVM imports
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# gaussian naive bayes imports
from sklearn.naive_bayes import GaussianNB

# gradient boosting imports
from sklearn.ensemble import GradientBoostingClassifier

from utils.process_audio_data import generate_target_csv, perform_PCA_testing, perform_PCA_training
from utils.consts import training
import numpy as np


def train_library_random_forest(training_data_directory: str):

    print('Starting random forest classifier')

    X_train, y_train, X_test, y_test, kaggle_data = combined_data_processing(training_data_directory, 'data/test')

    print(f'Shape of X_train data: {X_train.shape}')
    print(f'Shape of X_test data: {X_test.shape}')
    print(f'Shape of y_train data: {y_train.shape}')
    print(f'Shape of y_test data: {y_test.shape}')

    # create classifier model for training data
    print('Creating classifier...')
    classifier = RandomForestClassifier(max_depth=5)
    classifier.fit(X_train, y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


def train_library_svm(training_data_directory: str):

    print('Starting SVM classifier')

    X_train, y_train, X_test, y_test, kaggle_data = combined_data_processing(training_data_directory, 'data/test')

    print(f'Shape of X_train data: {X_train.shape}')
    print(f'Shape of X_test data: {X_test.shape}')
    print(f'Shape of y_train data: {y_train.shape}')
    print(f'Shape of y_test data: {y_test.shape}')   

    # create classifier model for training data
    print('Creating classifier...')
    classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    classifier.fit(X_train, y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


def train_library_naive_bayes(training_data_directory: str):

    print('Starting naive bayes classifier')

    X_train, y_train, X_test, y_test, kaggle_data = combined_data_processing(training_data_directory, 'data/test')

    print(f'Shape of X_train data: {X_train.shape}')
    print(f'Shape of X_test data: {X_test.shape}')
    print(f'Shape of y_train data: {y_train.shape}')
    print(f'Shape of y_test data: {y_test.shape}')

    # create classifier model for training data
    print('Creating classifier...')
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


def train_library_gradient_boosting(training_data_directory: str):

    print('Starting gradient boosting classifier')

    X_train, y_train, X_test, y_test, kaggle_data = combined_data_processing(training_data_directory, 'data/test')

    print(f'Shape of X_train data: {X_train.shape}')
    print(f'Shape of X_test data: {X_test.shape}')
    print(f'Shape of y_train data: {y_train.shape}')
    print(f'Shape of y_test data: {y_test.shape}')

    # create classifier model for training data
    print('Creating classifier...')
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    classifier.fit(X_train, y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


if __name__== '__main__':
    train_library_svm('data/train')
