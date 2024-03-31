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

    X_train, X_test, y_train, y_test = create_test_train_split(training_data_dir)
    

    y_training_one_hot, y_categories = generate_one_hot(y_train[0])

    W_trained = updated_gradient_descent(X_training=X_train, 
                                         Y_training=y_training_one_hot, 
                                         Y_categories=y_categories)
    
    weights_df = pd.DataFrame(W_trained, columns=X_train.columns)
    weights_df.to_csv(feature_file_dir + '/model4.csv')

    result_indexes = np.argmax(np.matmul(X_test, W_trained.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W_trained), axis=1)
    results = np.take(y_categories, result_indexes)
    testing_df = pd.DataFrame()
    testing_df['acutal'], testing_df['predicted'] = y_test, results
    testing_df.to_csv('./predicted_output.csv')
    print('-'*10, 'accuracy : ', metrics.accuracy_score(y_test, results), '-'*10)


if __name__ == '__main__':

    train_logistic_regression('data/train')


