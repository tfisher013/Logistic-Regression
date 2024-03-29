from typing import List
import os
import numpy as np
from utils.consts import *
from utils.process_audio_data import *
import math
import pandas as pd
import shutil
from sklearn import metrics
from numpy import linalg as LA


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

    # generate feature csv files
    print(f'the contents of {training_data_dir} are {os.listdir(training_data_dir)}')
    for class_dir in os.listdir(training_data_dir):
        class_dir_path = os.path.join(training_data_dir, class_dir)
        if os.path.isdir(class_dir_path) and class_dir.startswith('.') == False:
            generate_target_csv(class_dir_path)
        
    df_training = combine_files_save_to_one(training)
    df_training = df_training.sample(frac = 1)

    X_training =  df_training.drop(columns[-1], axis= 1).to_numpy()
    Y_training = df_training[columns[-1]].to_numpy()
    Y_training_one_hot, Y_categories = generate_one_hot(Y_training)

    W_Trained = updated_gradient_descent(X_training=X_training, 
                                         Y_training=Y_training_one_hot, 
                                         Y_categories=Y_categories)
    
    weights_df = pd.DataFrame(W_Trained, columns = columns[:-1])
    weights_df.to_csv(feature_file_dir + '/model.csv')

    result_indexes = np.argmax(np.matmul(X_training, W_Trained.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W_Trained), axis=1)
    results = np.take(Y_categories, result_indexes)
    print(Y_training, results)
    testing_df = pd.DataFrame()
    testing_df['acutal'],testing_df['predicted'] = Y_training, results
    testing_df.to_csv('./predicted_output.csv')
    print('-'*10, 'accuracy : ', metrics.accuracy_score(Y_training, results), '-'*10)


if __name__ == '__main__':

    train_logistic_regression('data/train')


