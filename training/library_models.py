# random forest imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

# SVM imports
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# gaussian naive bayes imports
from sklearn.naive_bayes import GaussianNB

# gradient boosting imports
from sklearn.ensemble import GradientBoostingClassifier

from utils.process_audio_data import *
import numpy as np


def library_model_hyperparameter_search():
    """ Performs a hyperparameter search over the 4 library ML functions
        used for a comparison to our logistic regression implementation.
    """

    # get training/testing data
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test= np.load('y_test.npy')

    # create k fold cross validation partitions
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    combined_train, combined_test = np.append(X_train, X_test, axis=0), np.append(y_train, y_test, axis=0)

    #### random forest hyperparameter search ####
    print('Random Forest')

    # test different split criterion
    for criterion in ['gini', 'entropy', 'log_loss']:
        # test different numbers of trees
        for n_estimators in [50, 75, 100, 125, 150]:

            classifier = RandomForestClassifier()

            # average balanced accuracy across k-folds
            total_acc = 0
            for train_idx, test_idx in skf.split(combined_train, combined_test):
                classifier.fit(combined_train[train_idx], combined_test[train_idx])
                total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

            print(f'  Parameter set: criterion={criterion}, n_estimators={n_estimators}')
            print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')

    #### SVM hyperparameter search ####
    print('\n\nSVM')

    # test different kernel coefficients
    for kernel_coefficient in ['auto', 'scale']:
        # test different kernel types
        for kernel in ['linear', 'poly', 'sigmoid']:

            classifier = SVC(kernel=kernel, gamma=kernel_coefficient)

            # average balanced accuracy across k-folds
            total_acc = 0
            for train_idx, test_idx in skf.split(combined_train, combined_test):
                classifier.fit(combined_train[train_idx], combined_test[train_idx])
                total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

            print(f'  Parameter set: kernel coefficient={kernel_coefficient}, kernel={kernel}')
            print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')

    #### NB hyperparameter search ####
    print('\n\nNaive Bayes')

    # test different smoothing values
    for var_smoothing in [1e-11, 1e-9, 1e-7]:

            classifier = GaussianNB(var_smoothing=var_smoothing)

            # average balanced accuracy across k-folds
            total_acc = 0
            for train_idx, test_idx in skf.split(combined_train, combined_test):
                classifier.fit(combined_train[train_idx], combined_test[train_idx])
                total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

            print(f'  Parameter set: var_smoothing={var_smoothing}')
            print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')

    #### GB hyperparameter search ####
    print('\n\nGradient Boosting')

    # test different GB criterion
    for criterion in ['friedman_mse', 'squared_error']:
        # test different loss metrics
        for learning_rate in [0.01, 0.1, 1.0]:
            # test different methods for determining number of max features
            for max_features in ['sqrt', 'log2']:

                classifier = GradientBoostingClassifier(criterion=criterion, 
                                                        learning_rate=learning_rate, 
                                                        max_features=max_features, 
                                                        random_state=0)
                
                # average balanced accuracy across k-folds
                total_acc = 0
                for train_idx, test_idx in skf.split(combined_train, combined_test):
                    classifier.fit(combined_train[train_idx], combined_test[train_idx])
                    total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

                print(f'  Parameter set: criterion={criterion}, learning_rate={learning_rate}, max_features={max_features}')
                print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')




def library_model_hyperparameter_search():
    """ Performs a hyperparameter search over the 4 library ML functions
        used for a comparison to our logistic regression implementation.
    """

    # get training/testing data
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_test= np.load('y_test.npy')

    # create k fold cross validation partitions
    skf = StratifiedKFold(n_splits=3, shuffle=True)
    combined_train, combined_test = np.append(X_train, X_test, axis=0), np.append(y_train, y_test, axis=0)

    #### random forest hyperparameter search ####
    print('Random Forest')

    # test different split criterion
    for criterion in ['gini', 'entropy', 'log_loss']:
        # test different numbers of trees
        for n_estimators in [50, 75, 100, 125, 150]:

            classifier = RandomForestClassifier()

            # average balanced accuracy across k-folds
            total_acc = 0
            for train_idx, test_idx in skf.split(combined_train, combined_test):
                classifier.fit(combined_train[train_idx], combined_test[train_idx])
                total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

            print(f'  Parameter set: criterion={criterion}, n_estimators={n_estimators}')
            print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')

    #### SVM hyperparameter search ####
    print('\n\nSVM')

    # test different kernel coefficients
    for kernel_coefficient in ['auto', 'scale']:
        # test different kernel types
        for kernel in ['linear', 'poly', 'sigmoid']:

            classifier = SVC(kernel=kernel, gamma=kernel_coefficient)

            # average balanced accuracy across k-folds
            total_acc = 0
            for train_idx, test_idx in skf.split(combined_train, combined_test):
                classifier.fit(combined_train[train_idx], combined_test[train_idx])
                total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

            print(f'  Parameter set: kernel coefficient={kernel_coefficient}, kernel={kernel}')
            print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')

    #### NB hyperparameter search ####
    print('\n\nNaive Bayes')

    # test different smoothing values
    for var_smoothing in [1e-11, 1e-9, 1e-7]:

            classifier = GaussianNB(var_smoothing=var_smoothing)

            # average balanced accuracy across k-folds
            total_acc = 0
            for train_idx, test_idx in skf.split(combined_train, combined_test):
                classifier.fit(combined_train[train_idx], combined_test[train_idx])
                total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

            print(f'  Parameter set: var_smoothing={var_smoothing}')
            print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')

    #### GB hyperparameter search ####
    print('\n\nGradient Boosting')

    # test different GB criterion
    for criterion in ['friedman_mse', 'squared_error']:
        # test different loss metrics
        for learning_rate in [0.01, 0.1, 1.0]:
            # test different methods for determining number of max features
            for max_features in ['sqrt', 'log2']:

                classifier = GradientBoostingClassifier(criterion=criterion, 
                                                        learning_rate=learning_rate, 
                                                        max_features=max_features, 
                                                        random_state=0)
                
                # average balanced accuracy across k-folds
                total_acc = 0
                for train_idx, test_idx in skf.split(combined_train, combined_test):
                    classifier.fit(combined_train[train_idx], combined_test[train_idx])
                    total_acc += balanced_accuracy_score(combined_test[test_idx], 
                                                     classifier.predict(combined_train[test_idx]))

                print(f'  Parameter set: criterion={criterion}, learning_rate={learning_rate}, max_features={max_features}')
                print(f'    Balanced accuracy: {total_acc / skf.get_n_splits()}')


def train_library_random_forest(training_data_directory: str):
    """ Trains sklearn's random forest model on our data pipeline and
        shows the resulting accuracy

        Parameters:
            training_data_directory: the path to the directory containing
                training data for the model
    """

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
    """ Trains sklearn's SVM model on our data pipeline and
        shows the resulting accuracy

        Parameters:
            training_data_directory: the path to the directory containing
                training data for the model
    """

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
    """ Trains sklearn's naive bayes model on our data pipeline and
        shows the resulting accuracy

        Parameters:
            training_data_directory: the path to the directory containing
                training data for the model
    """

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
    """ Trains sklearn's gradient boosting model on our data pipeline and
        shows the resulting accuracy

        Parameters:
            training_data_directory: the path to the directory containing
                training data for the model
    """

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
    """ Main function for testing
    """
    
    train_library_svm('data/train')
