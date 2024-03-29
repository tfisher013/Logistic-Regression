import numpy as np
def validate_model( W : np.array  , X_testing : np.array , Weights_to_target_map : dict):
    """
    This function is used to validate a given model with weights : W

    parameters :
        X_testing: a matrix of testing data where each row represents  a testing instance

        W : a matrix of weights where each row represents a weight vector that is good enought to predict a class K 
         
    
    """
    targets = np.array()
    for testing_instance in X_testing:
        aggregates = W * np.transpose(testing_instance)
        max_index = np.argmax(aggregates)
        targets.append(Weights_to_target_map.get(max_index))
    return targets







