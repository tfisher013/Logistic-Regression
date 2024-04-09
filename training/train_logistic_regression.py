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
from utils.process_audio_data import normalize_columns,normalize_row
from validation.validate import validate_model
import sys

def vectorized_probability(matrix: np.array, mode : str)-> np.array :
    matrix = matrix - np.max(matrix , axis = 1 , keepdims=True )
    matrix = np.exp(matrix)
    return matrix/np.sum(matrix , axis = 1 , keepdims=True)


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

    while model_error > epsilon and iteration_count < max_iterations:

        X_cross_W = np.matmul(X_training, W.T)
        matrix_of_samples_and_weights_product = vectorized_probability(X_cross_W, 'others')

        step_size_boost = 1 + (iteration_count // 20000)

        gradients = (np.matmul((Y_training - matrix_of_samples_and_weights_product).T, X_training))

        W += (1/Y_training.shape[0]) * step_size_boost * eta * gradients - eta * lambda_hyperparameter * W
 
        # compute model error as sum of squared errors between Y and Y_hat
        X_cross_W = np.matmul(X_training, W.T)
        matrix_of_samples_and_weights_product = vectorized_probability(X_cross_W, 'others') 
        model_error = np.sum((Y_training - matrix_of_samples_and_weights_product) ** 2, axis=None)

        iteration_count += 1

        print("-"*10, f"Loss : {model_error} ", "-"*10, f"Iteration  {iteration_count}", "-"*10, f"{max(1, int(iteration_count // 20000))}")

    return W


def train_logistic_regression(training_data_dir: str):
    """ Trains a logistic regression model using the training data in the provided
            directory

        Parameters:
            training_data_dir: path to directory which contains training instances 
                each organized into separate directories by class

    """

    X_train , y_train , X_test , y_test , kaggle ,  mean_variance_train , mean_variance_test , feature_kaggle = combined_data_processing('data/train' , 'data/test')
    X_train , X_test , kaggle = mean_variance_train , mean_variance_test , feature_kaggle 

    #X_train = np.load('X_train.npy')
    #X_test = np.load('X_test.npy')
    #kaggle = np.load('X_kaggle.npy')
    #y_train = np.load('y_train.npy')
    #y_test= np.load('y_test.npy')

    X_train = np.append(X_train, np.ones(X_train.shape[0]).reshape(-1, 1), 1) 
    X_test = np.append(X_test  , np.ones(X_test.shape[0]).reshape(-1, 1), 1)
    kaggle = np.append(kaggle  , np.ones(kaggle.shape[0]).reshape(-1, 1), 1)

    y_training_one_hot, y_categories = generate_one_hot(y_train.reshape(-1,1))

    W_trained = updated_gradient_descent(X_training=X_train, 
                                         Y_training=y_training_one_hot, 
                                         Y_categories=y_categories)
    
    weights_df = pd.DataFrame(W_trained, columns=[i for i in range(W_trained.shape[1])])
    weights_df.to_csv(model_dir + '/model4.csv')

    result_indexes = np.argmax(np.matmul(X_test, W_trained.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W_trained), axis=1)
    results = np.take(y_categories, result_indexes)

    # for testing on train/test split
    print('-'*10, 'accuracy : ', metrics.accuracy_score(y_test, results), '-'*10)
    
    # create predictions file
    validate_model(W = W_trained , 
                   X_test = kaggle , 
                   y_categories = y_categories  , 
                   file_names= os.listdir('data/test'))

    

if __name__ == '__main__':

    train_logistic_regression('data/train')


