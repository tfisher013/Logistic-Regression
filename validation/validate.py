import numpy as np
from utils.consts import lambda_hyperparameter
import pandas as pd
from numpy import linalg as LA

def validate_model( W : np.array  , X_test: np.array , y_categories , file_names : np.array ):
    """
    This function is used to validate a given model with weights : W

    parameters :
        X_testing: a matrix of testing data where each row represents  a testing instance

        W : a matrix of weights where each row represents a weight vector that is good enought to predict a class K 
         
        y_categories : a array of ordered labels extracted from onehotencoder

    
    """
    print( X_test.shape , W.shape)

    result_indexes = np.argmax(np.matmul(X_test ,  W.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W), axis=1)
    results = np.take(y_categories, result_indexes)
    predicted = pd.DataFrame()
    predicted['id'] = file_names
    predicted['class'] = results
    print(predicted)
    predicted.to_csv('predicted_results.csv' , index = False)
    








