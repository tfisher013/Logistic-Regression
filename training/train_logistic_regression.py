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
    # print(matrix)n
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

    while model_error > epsilon  and iteration_count < max_iterations:
        # create matrix of prediction probabilities and normalize by columns
        X_cross_W = np.matmul(X_training, W.T)
        # X_cross_W = (array=X_cross_W)
        # X_cross_W[: , :-1] = 0
        matrix_of_samples_and_weights_product = vectorized_probability(X_cross_W, 'others')
        # matrix_of_samples_and_weights_product /= np.max(matrix_of_samples_and_weights_product, axis=1, keepdims=True)
        # print(matrix_of_samples_and_weights_product.shape , (1 - np.sum(matrix_of_samples_and_weights_product[:, :-1], axis=1)).shape)
        # matrix_of_samples_and_weights_product[:, -1] = 1 - np.sum(matrix_of_samples_and_weights_product[:, :-1], axis=1)

        gradients = (np.matmul((Y_training - matrix_of_samples_and_weights_product).T, X_training))
        W += (1/Y_training.shape[0]) * eta * gradients - eta * lambda_hyperparameter * W
 
        # compute model error as sum of squared errors between Y and Y_hat
        X_cross_W = np.matmul(X_training, W.T)
        # X_cross_W /= np.max(X_cross_W, axis=1, keepdims=True)
        matrix_of_samples_and_weights_product = vectorized_probability(X_cross_W, 'others') 
        # matrix_of_samples_and_weights_product[:, -1] = 1 - np.sum(matrix_of_samples_and_weights_product[:, :-1], axis=1)
        #prediction_matrix = np.multiply(Y_training, X_cross_W) - np.log(1 + np.exp(X_cross_W))
        #prediction_matrix /= np.max(prediction_matrix, axis=1, keepdims=True)
        model_error = np.sum((Y_training - matrix_of_samples_and_weights_product) ** 2, axis=None)

        iteration_count += 1

        print("-"*10, f"Loss : {model_error} ", "-"*10, f"Iteration  {iteration_count}")

    return W


def train_logistic_regression(training_data_dir: str):
    """ Trains a logistic regression model using the training data in the provided
            directory

        Parameters:
            training_data_dir: path to directory which contains training instances 
                each organized into separate directories by class

    """

    # 1. create a dataset of all test sample
    # X, Y = create_combined_df(path)
    X_train , y_train , X_test , y_test , kaggle ,  mean_variance_train , mean_variance_test , feature_kaggle = combined_data_processing('data/train' , 'data/test')
    # 2. create train/test splits
    # X_train, X_test, y_train, y_test = ...
    X_train , X_test , kaggle = mean_variance_train , mean_variance_test , feature_kaggle
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify= Y) 

    # 3. perform processing on each 
    # X_train_processed = process_training(X_train)

    X_train = np.append(X_train, np.ones(X_train.shape[0]).reshape(-1, 1), 1) 
    X_test = np.append(X_test  , np.ones(X_test.shape[0]).reshape(-1, 1), 1)
    
    kaggle = np.append( kaggle , np.ones(kaggle.shape[0]).reshape(-1, 1), 1)
    # X_train = np.array([[-1, -2],
    #                    [-5, -6],
    #                    [-7, -10],
    #                    [-1, 5],
    #                    [-5, -5],
    #                    [-4, 1]])
    
    # y_train = np.array([0, 0, 0, 1, 1, 1]).reshape(-1, 1)

    # X_test = np.array([[5, 0], [10, 2], [-10, -5]])
    # y_test = np.array([0, 0, 1]).reshape(-1, 1)


    # X_combined_train , Y_train = generate_target_csv(training_data_dir)
    # X_train, X_test, y_train, y_test = train_test_split(X_combined_train, Y_train, test_size=0.3, stratify= Y_train)    

    y_training_one_hot, y_categories = generate_one_hot(y_train.reshape(-1,1))

    W_trained = updated_gradient_descent(X_training=X_train, 
                                         Y_training=y_training_one_hot, 
                                         Y_categories=y_categories)
    
    weights_df = pd.DataFrame(W_trained, columns=[i for i in range(W_trained.shape[1])])
    weights_df.to_csv(feature_file_dir + '/model4.csv')

    result_indexes = np.argmax(np.matmul(X_test, W_trained.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W_trained), axis=1)
    results = np.take(y_categories, result_indexes)

    # for testing on train/test split
    testing_df = pd.DataFrame()
    testing_df['acutal'], testing_df['predicted'] = y_test, results
    testing_df.to_csv('./predicted_output.csv')
    print('-'*10, 'accuracy : ', metrics.accuracy_score(y_test, results), '-'*10)
    
    # create predictions file
    validate_model(W = W_trained , X_test = kaggle , y_categories = y_categories  , file_names= os.listdir('data/test'))

    

if __name__ == '__main__':

    train_logistic_regression('data/train')


