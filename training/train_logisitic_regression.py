import numpy as np

def p_hat(X_i: np.array, W_i: np.array) -> float:
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

    positive_pred_frac = 1 / (1 + np.exp(np.dot(W_i, X_i.T)))

    return positive_pred_frac[positive_pred_frac == 1] / len(positive_pred_frac)