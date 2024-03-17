from typing import List
import os
import time
import numpy as np
from utils.consts import *
from utils.process_audio_data import generate_target_csv


def p_hat(X: np.array, W_i: np.array) -> float:
    """ Computes the predicted probability of a positive class
            instance given the provided model weights and a
            training instance

        Parameters:
            X_i: one instance vector of training data which
                is associated with the positive class
            W_i: the weights vector for the class being tested

        Returns:
            the predicted probability of a positive occurrence
                given the training instance and weight vectors

    """

    positive_pred_frac = 1 / (1 + np.exp(np.dot(X, W_i.T)))

    return len(positive_pred_frac[positive_pred_frac > 0]) / len(positive_pred_frac)


def train_logistic_regression(training_data_dir: str):
    """ Trains a logistic regression model using the training data in the provided
            directory

        Parameters:
            training_data_dir: path to directory which contains training instances 
                each organized into separate directories by class

    """

    # determine the number of classes
    class_labels = []
    for class_dir in os.listdir(training_data_dir):
        if os.path.isdir(class_dir):
            class_labels.append(class_dir)

    # initialize weight matrix to zeros
    W = np.zeros((len(class_labels), num_features))

    # train one row of W per class
    print(f'the contents of {training_data_dir} are {os.listdir(training_data_dir)}')
    for idx, class_dir in enumerate(os.listdir(training_data_dir)):
        class_dir_path = os.path.join(training_data_dir, class_dir)
        if os.path.isdir(class_dir_path) and class_dir.startswith('.') == False:
        
            # generate CSV file using specified directory
            class_csv_file = generate_target_csv(class_dir_path)
        
            # read CSV file into training instance matrix
            X_train_class = np.genfromtxt(class_csv_file, delimiter=',', skip_header=1)
        
            # use gradient descent to set values of this row of W
            W[idx] = gradient_descent(X_train_class)
    
    # save model to file after training
    model_name = 'model' + str(int(time.time()))
    if not os.path.isdir('models'):
        os.mkdir('models')
    np.savetxt(os.path.join(model_dir, model_name), W)
    


def gradient_descent(X_training: np.array) -> np.array:
    """ Uses gradient descent to train a weights matrix using logistic
            regression

        Parameters:
            X_training: a matrix of training data where each row represents
                a training instance

        Returns:
            a 1D numpy array representing the trained weights for the
                particular class being trained for with this function call
    """

    num_features = X_training.shape[1]

    # initialize weight vector to random values
    w = np.random.rand(num_features)

    target_model_error = float('inf')
    while target_model_error > error_threshold:

        w += eta * (1 - p_hat(X_training, w)) * np.sum(X_training, axis=0)

        target_model_error = 0
        for i in X_training:
            target_model_error += (np.dot(i, w)) ** 2

    return w

if __name__ == '__main__':

    train_logistic_regression('../data/train')


