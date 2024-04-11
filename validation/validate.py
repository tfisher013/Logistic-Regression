from matplotlib import cm
import numpy as np
from utils.consts import lambda_hyperparameter
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def validate_model( W : np.array  , X_test: np.array , y_categories , file_names : np.array ):
    """
    This function is used to validate a given model with weights : W

    parameters :
        X_testing: a matrix of testing data where each row represents  a testing instance

        W : a matrix of weights where each row represents a weight vector that is good enought to predict a class K 
         
        y_categories : a array of ordered labels extracted from onehotencoder

        file_names: the names of the files containing the testing data
    
    """
    print( X_test.shape , W.shape)
    files = []
    for file in file_names:
        if file.endswith('.au'):
            files.append(file)
    file_names = files
    result_indexes = np.argmax(np.matmul(X_test ,  W.T) - 
                               ((lambda_hyperparameter / 2) ** 2) * LA.norm(W), axis=1)
    results = np.take(y_categories, result_indexes)
    predicted = pd.DataFrame()
    predicted['id'] = file_names
    predicted['class'] = results
    print(predicted)
    predicted.to_csv('kaggle_predictions.csv' , index = False)
    

def plot_confusion_matrix( actual_results : np.array , predicted_results : np.array , classes : np.array ):
    """
    This function is used to validate a given model with weights : W

    parameters :
        actual_results : a array of actual results 

        predicted_results : array of predicted results  
         
        classes : a array of ordered labels extracted from onehotencoder

    
    """

    title = 'Confusion Matix' #title of the plot that is to be generated

    conf_matrix = confusion_matrix(actual_results , predicted_results)
    figure , axis = plt.subplots(figsize = (7,5))
    cmap=plt.cm.Blues #colormap that is used to generate the confusion matrix 
    im = axis.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    axis.figure.colorbar(im, ax=axis)
    
    # We want to show all targets and label them with the respective list entries
    axis.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Aims to rotate the  labels and set their position.
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations with customized colors
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            axis.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
                    
    figure.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    # plt.show()
    
    return conf_matrix






