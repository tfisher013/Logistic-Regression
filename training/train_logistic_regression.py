from typing import List
import os
import numpy as np
from utils.consts import *
from utils.process_audio_data import *
import math
import pandas as pd
import shutil
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
from sklearn.model_selection import train_test_split
from utils.process_audio_data import normalize_columns


def vectorized_probability(matrix: np.array, mode : str)-> np.array :
    if mode == others:
        return np.vectorize(lambda z: math.exp(z))(matrix)
    else:
        return 1


def updated_gradient_descent(X_training: np.array, Y_training: np.array, Y_categories : np.array) -> np.array:
    """ Performs gradient descent using the provided training data, split between feature
            values and associated classe0.s

        Parameters:
            X_training: a matrix where each row represents a training instance without
                an associated class being included
            Y_training: a matrix with each row representing the class associated with
                the corresponding training instance in X_training
            Y_categories: a one-hot-encoding of Y_training

        Returns:
            a matrix of weights representing the trained model
    """

    W: np.array = np.random.rand(Y_categories.shape[0], X_training.shape[1])
    model_error, iteration_count = np.inf, 0

    while abs(model_error) > epsilon and iteration_count < max_iterations:

        matrix_of_samples_and_weights_product = vectorized_probability(
            pd.DataFrame(np.matmul(X_training, W.T)).to_numpy(), 'others')

        gradients = (np.matmul((Y_training - matrix_of_samples_and_weights_product).T, X_training))
        W += eta * gradients - eta * lambda_hyperparameter * W

        model_error = np.sum(gradients)
        model_error, iteration_count = np.sum(model_error), iteration_count+1

        print("-"*10, f"Loss : {np.sum(model_error) } ", "-"*10, f"Iteration  {iteration_count}")

    return W


def train_logistic_regression(training_data_dir: str):
    """ Trains a logistic regression model using the training data in the provided
            directory

        Parameters:
            training_data_dir: path to directory which contains training instances 
                each organized into separate directories by class

    """

    # determine the number of classes
    if os.path.exists(feature_file_dir):
        shutil.rmtree(feature_file_dir)
    class_labels = []
    for class_dir in os.listdir(training_data_dir):
        if os.path.isdir(class_dir):
            class_labels.append(class_dir)

    # create dataframes to hold the combined train/test data
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame() 

    print(f'the contents of {training_data_dir} are {os.listdir(training_data_dir)}')
    for class_dir in os.listdir(training_data_dir):
        class_dir_path = os.path.join(training_data_dir, class_dir)
        if os.path.isdir(class_dir_path) and class_dir.startswith('.') == False:

            # convert feature data to CSV and DataFrame formats
            feature_data_file = generate_target_csv(class_dir_path)
            feature_df = pd.read_csv(feature_data_file, header=None)

            print(feature_df)

            # perform train test split on feature data
            feature_X_train, feature_X_test, feature_y_train, feature_y_test = train_test_split(
                feature_df.drop(feature_df.columns[-1], axis=1), 
                feature_df[feature_df.columns[-1]], 
                test_size=0.2,
                stratify=feature_df[feature_df.columns[-1]])
            
            # append feature train/test data to entire train/test data
            X_train = pd.concat([X_train, feature_X_train], axis=0, ignore_index=True)
            X_test = pd.concat([X_test, feature_X_test], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, feature_y_train], axis=0, ignore_index=True)
            y_test = pd.concat([y_test, feature_y_test], axis=0, ignore_index=True)

    print(f'1. dimension of X_train before standardization and PCA: {X_train.shape}')
    print(X_train)

    # standardize columns with distributions generated
    # by training set
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train))
    X_test = pd.DataFrame(sc.transform(X_test))

    print(f'2. dimension of X_train after standardization: {X_train.shape}')
    print(X_train)
            
    # perform PCA using features selected from training
    # set
    pca = PCA(n_components=0.8)
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_test = pd.DataFrame(pca.transform(X_test))

    # normalize data columnwise
    X_train = pd.DataFrame(normalize_columns(X_train))
    X_test = pd.DataFrame(normalize_columns(X_test))

            
    print(f'3. dimension of X_train after standardization and PCA: {X_train.shape}')
    print(X_train)
    print(y_train)

    # add column of ones to X_train and X_test
    X_train[len(X_train.columns)] = 1
    X_test[len(X_test.columns)] = 1

    ####### Feature processing complete, proceed as before #######
    

    y_training_one_hot, y_categories = generate_one_hot(y_train[0])

    W_trained = updated_gradient_descent(X_training=X_train, 
                                         Y_training=y_training_one_hot, 
                                         Y_categories=y_categories)
    
    weights_df = pd.DataFrame(W_trained, columns=X_train.columns)
    weights_df.to_csv(feature_file_dir + '/model4.csv')

    result_indexes = np.argmax(np.matmul(X_train, W_trained.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W_trained), axis=1)
    results = np.take(y_categories, result_indexes)
    testing_df = pd.DataFrame()
    testing_df['acutal'], testing_df['predicted'] = y_train, results
    testing_df.to_csv('./predicted_output.csv')
    print('-'*10, 'accuracy : ', metrics.accuracy_score(y_train, results), '-'*10)


if __name__ == '__main__':

    train_logistic_regression('data/train')


