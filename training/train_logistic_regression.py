import os
import numpy as np
from utils.consts import *
from utils.process_audio_data import *
import pandas as pd
from sklearn import metrics
from numpy import linalg as LA
from validation.validate import validate_model


def vectorized_probability(matrix: np.array) -> np.array:
    """ Function used to compute the softmax of the provided matrix, intended
        to be passed as argument to np.vectorize() for increased speed

        Parameters:
            matrix: a numpy-represented matrix

        Returns:
            the softmax probabilty matrix computed from the provided matrix
    """
    
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    matrix = np.exp(matrix)
    return matrix/np.sum(matrix, axis=1, keepdims=True)


def gradient_descent(X_training: np.array, Y_training: np.array, Y_categories : np.array) -> np.array:
    """ Performs gradient descent using the provided training data, split between feature
        values and associated classe0.s

        Parameters:
            X_training: a matrix where each row represents a training instance without
                an associated class column being included
            Y_training: a matrix with a single column where value represents the true class
                associated with the corresponding training instance in X_training
            Y_categories: a transformation of Y_training where all values have been one-
                hot-encoded

        Returns:
            a matrix of weights representing the trained model
    """

    W: np.array = np.random.rand(Y_categories.shape[0], X_training.shape[1])
    model_error, iteration_count = np.inf, 0

    while model_error > epsilon  and iteration_count < max_iterations:

        X_cross_W = np.matmul(X_training, W.T)
        matrix_of_samples_and_weights_product = vectorized_probability(X_cross_W, 'others')

        # use a step size "booster" which grows with certain iteration jumps
        step_size_boost = max(1, iteration_count // 20000)

        gradients = (np.matmul((Y_training - matrix_of_samples_and_weights_product).T, X_training))
        W += (1/Y_training.shape[0]) * step_size_boost * eta * gradients - eta * lambda_hyperparameter * W
 
        # compute model error as sum of squared errors between Y and Y_hat
        X_cross_W = np.matmul(X_training, W.T)
        matrix_of_samples_and_weights_product = vectorized_probability(X_cross_W, 'others') 
        model_error = np.sum((Y_training - matrix_of_samples_and_weights_product) ** 2, axis=None)

        iteration_count += 1

        print("-"*10, f"Loss : {model_error} ", "-"*10, f"Iteration  {iteration_count}", "-"*10, f"step boost: {step_size_boost}")

    return W


def train_logistic_regression(training_data_dir: str):
    """ Trains a logistic regression model using the training data in the provided
        directory

        Parameters:
            training_data_dir: path to directory which contains training instances 
                each organized into separate directories by class

    """

    # read and process audio data
    X_train , y_train , X_test , y_test , kaggle ,  mean_variance_train , mean_variance_test , feature_kaggle = combined_data_processing('data/train' , 'data/test')
    X_train , X_test , kaggle = mean_variance_train , mean_variance_test , feature_kaggle

    # replace the two lines above with these if the data has been written to  file 
    #X_train = np.load('X_train.npy')
    #X_test = np.load('X_test.npy')
    #kaggle = np.load('X_kaggle.npy')
    #y_train = np.load('y_train.npy')
    #y_test= np.load('y_test.npy')
    
    # append a colunm of 1's to all training matrices
    X_train = np.append(X_train, np.ones(X_train.shape[0]).reshape(-1, 1), 1) 
    X_test = np.append(X_test  , np.ones(X_test.shape[0]).reshape(-1, 1), 1)
    kaggle = np.append( kaggle , np.ones(kaggle.shape[0]).reshape(-1, 1), 1)
 
    # create one-hot-encoding representation of classes
    y_training_one_hot, y_categories = generate_one_hot(y_train.reshape(-1,1))

    # perform gradient descent to train W matrix
    W_trained = gradient_descent(X_training=X_train, 
                                         Y_training=y_training_one_hot, 
                                         Y_categories=y_categories)

    # write weight matrix to file
    weights_df = pd.DataFrame(W_trained, columns=[i for i in range(W_trained.shape[1])])
    weights_df.to_csv(feature_file_dir + '/model4.csv')

    # generate accuracy on test split
    result_indexes = np.argmax(np.matmul(X_test, W_trained.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W_trained), axis=1)
    results = np.take(y_categories, result_indexes)
    testing_df = pd.DataFrame()
    testing_df['acutal'], testing_df['predicted'] = y_test, results
    testing_df.to_csv('./predicted_output.csv')
    print('-'*10, 'accuracy : ', metrics.accuracy_score(y_test, results), '-'*10)
    
    # create predictions file for kaggle
    validate_model(W=W_trained, X_test=kaggle, y_categories=y_categories, file_names=os.listdir('data/test'))
    

if __name__ == '__main__':
    """ Main function for testing
    """

    train_logistic_regression('data/train')


