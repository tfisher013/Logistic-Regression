from typing import List
import os
import time
import numpy as np
from utils.consts import *
from utils.process_audio_data import generate_target_csv



def classify(x: np.array, W: np.array, class_values: np.array) -> np.array:
    """ Returns the class to which the model predicts the provided instance
            belongs

        Parameters:
            x: a vector of features represented as a 1D array
            W: the matrix of weights represented as a 2D array
            class_values: a matrix containing each class in one-hot-encoding
                as a row vector

        Returns:
            the predicted class in one-hot-encoding
    """

    predicted_probabilities = np.exp(np.dot(x, W))

    # replace the last entry with 1 - the sum over the other entries;
    # this is an invariant of our predictions vector that it sums to 1
    predicted_probabilities[-1] = 1 - np.sum(predicted_probabilities[:-1])

    # return the predicted class (as a one-hot-encoding vector), gained from
    # the index of the max value in the vector of predicted probabilities
    return class_values(np.argmax(predicted_probabilities))


def generate_class_predictions(X: np.array, W: np.array, class_values: np.array) -> np.array:
    """ Generates a vector of predictions for each row of the provided X

        Parameters:
            X: a matrix of instances, with each row representing a single instance
            W: the model weight matrix
            class_values: a matrix containing each class in one-hot-encoding
                as a row vector

        Returns
            a matrix of predicted classes for each instance in X using model weights
                W where each row represents a predicted class in one-hot-encoding for
                the corresponding entry in X in one-hot-encoding
    """

    # generate a matrix of predictions, where each row contains a vector of length
    # (# classes) and each element in the row vector represents the probability of 
    # that vector in X belong to a specific class
    prediction_matrix = np.dot(X, W)

    # replace last column of prediction matrix with 1 - sum(row elements)
    prediction_matrix[:,-1] = 1 - prediction_matrix[:,:-1].sum(axis=1)

    # return the class for each instance by using the column index of the max
    # value per row as a row index into the class_values matrix
    return np.take(class_values, np.argmax(prediction_matrix, axis=0))


def updated_gradient_descent(X_training: np.array, Y_training: np.array, class_values: np.array) -> np.array:
    """ Performs gradient descent using the provided training data, split between feature
            values and associated classes

        Parameters:
            X_training: a matrix where each row represents a training instance without
                an associated class being included
            Y_training: a matrix with each row representing the class associated with
                the corresponding training instance in X_training
            class_values: a vector containing each unique class

        Returns:
            a matrix of weights representing the trained model
    """

    W: np.array = np.random.rand()
    model_error = np.inf

    # generate a vector which transforms the vector of true classes for the
    # training data into a vector of each true class' index
    true_class_indices = np.searchsorted(class_values, Y_training)

    while model_error > epsilon:

        # generate predictions for each training instance using current weight matrix
        prediction_matrix = np.dot(X_training, W.T)

         # replace last column of prediction matrix with 1 - sum(row elements)
        prediction_matrix[:,-1] = 1 - prediction_matrix[:,:-1].sum(axis=1)

        # select 1 prediction from each row of the prediction_matrix such that the selected
        # prediction is that for the class associated with the training instance
        class_probabilities = prediction_matrix[np.arange(len(true_class_indices)), true_class_indices.ravel()]

        # since each probability is now the probability for the correct class of that
        # training instance, we can subtract all probabilities from 1 to get the error
        class_probabilities_error = 1 - class_probabilities

        # multiply the error by X to construct the gradient
        W += eta * (np.multiply(X_training.T, class_probabilities_error).T, - np.multiply(lambda_hyperparameter, W))

        # recompute error - how to do this? conditional data log likelihood?
        # TODO compute the error over the entire model and convert it to a float,
        #   see page 11 of the LR reading from Tom Mitchell for one idea of how
        #   to do this

    return W


########################## OLD IMPLEMENTATION BELOW THIS LINE ##########################


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
    while target_model_error > epsilon:

        w += eta * (1 - p_hat(X_training, w)) * np.sum(X_training, axis=0)

        target_model_error = 0
        for i in X_training:
            target_model_error += (np.dot(i, w)) ** 2

    return w

if __name__ == '__main__':

    train_logistic_regression('../data/train')


