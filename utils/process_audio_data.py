import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from utils.consts import *
import pandas as pd
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
import random

def combined_data_processing() -> tuple[np.array, np.array]:
    """ Processes both training and kaggle data at once. Splits training data into train and test splits.
        All train/test and kaggle data undergo standardization, PCA, and normalization.

        Returns:

    """

    # define the featureset functions
    featureset_functions = [librosa.feature.zero_crossing_rate,
                            librosa.feature.mfcc, 
                            librosa.feature.chroma_stft, 
                            librosa.feature.chroma_cqt,
                            librosa.feature.spectral.spectral_contrast,
                            librosa.feature.chroma_cens,
                            librosa.feature.tonnetz]
    
    # create empty arrays to hold training and kaggle datasets
    training_matrix_train = np.array([[]])
    training_matrix_test = np.array([[]])
    kaggle_matrix = np.array([[]])
    mean_variance_train = np.array([[]])
    mean_variance_kaggle = np.array([[]])
    mean_variance_test = np.array([[]])

    # define empty array to hold the labels for the training matrix
    training_matrix_labels = []

    # define which row indices will be for training vs. testing; this needs to be done
    # so the test train splits are for the same samples across all featuresets
    num_classes = 10
    num_samples_per_class = 90
    test_size = 0.3
    test_sample_indices = []
    for class_idx in range(1, num_classes + 1):
        new_indices = random.sample(range((class_idx - 1) * num_samples_per_class, class_idx * num_samples_per_class), int(test_size * num_samples_per_class))
        test_sample_indices.extend(new_indices)
    test_sample_indices = np.array(test_sample_indices)

    # iterate over each featureset function
    for featureset_idx, featureset_function in enumerate(featureset_functions):

        print(f'Beginning featureset {featureset_idx + 1} of {len(featureset_functions)} ({str(featureset_function)})...')

        # define empty arrays to hold just the features for this featureset
        feature_training_matrix = np.array([[]])
        feature_kaggle_matrix = np.array([[]])
        feature_mean_train_matrix = np.array([[]])
        feature_mean_kaggle_matrix = np.array([[]])

        # populate feature training matrix
        for class_dir in os.listdir(training_data_path):
            class_dir_path = os.path.join(training_data_path, class_dir)
            if os.path.isdir(class_dir_path) and class_dir.startswith('.') == False:

                print('  Beginning processing of class ' + class_dir)

                max_cols = 0

                # iterate over each file in the class
                for training_file in os.listdir(class_dir_path):
                    training_file_path = os.path.join(class_dir_path, training_file)
                    if os.path.isfile(training_file_path) and training_file.startswith('.') == False:

                        audio_data, sample_rate = librosa.load(training_file_path , res_type='kaiser_best')

                        # use a try/except to handle differences in kwargs between
                        # featureset functions
                        featureset_feature_matrix = np.array([[]])
                        try:
                            featureset_feature_matrix = featureset_function(y=audio_data, sr=sample_rate)
                        except TypeError:
                            featureset_feature_matrix = featureset_function(y=audio_data)

                        # trim columns as necessary and append flattened matrix row-wise
                        if feature_training_matrix.size == 0:
                            feature_mean_train_matrix = np.mean(featureset_feature_matrix, axis=1)
                            feature_mean_train_matrix = np.append(feature_mean_train_matrix,
                                                                  np.var(featureset_feature_matrix, axis=1))
                            feature_mean_train_matrix = np.append(feature_mean_train_matrix,
                                                                  np.max(featureset_feature_matrix, axis=1))
                            feature_mean_train_matrix = np.append(feature_mean_train_matrix,
                                                                  np.min(featureset_feature_matrix, axis=1))
                            feature_mean_train_matrix = np.squeeze(feature_mean_train_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T
                            feature_training_matrix = np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T
                        else:
                            #prasanth
                            dummy = np.append(np.mean(featureset_feature_matrix , axis = 1 , keepdims = True) , 
                                                np.append(np.var(featureset_feature_matrix , axis = 1 , keepdims = True), 
                                                                        np.append( np.max(featureset_feature_matrix , axis = 1 , keepdims = True), 
                                                                                                np.min(featureset_feature_matrix, axis = 1 , keepdims=True) , axis = 1  ) , axis = 1  ) , axis = 1 )
                            dummy = np.squeeze(dummy.flatten('F').reshape(1, -1)).reshape(-1,1).T
                            feature_mean_train_matrix = np.append(feature_mean_train_matrix , dummy , axis = 0)
                            
                            #prasanth


                            max_cols = min(featureset_feature_matrix.size, feature_training_matrix.shape[1])

                            feature_training_matrix = np.append(
                                feature_training_matrix[:, :max_cols],
                                np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T[:, :max_cols],
                                axis=0
                            )

                        if featureset_idx == 0:
                            training_matrix_labels.append(class_dir)

        # populate kaggle feature matrix
        for kaggle_file in os.listdir(testing_data_path):
            kaggle_file_path = os.path.join(testing_data_path, kaggle_file)
            if os.path.isfile(kaggle_file_path) and kaggle_file_path.startswith('.') == False:

                # ignore text file
                if os.path.splitext(kaggle_file)[1] != '.au':
                    continue

                audio_data, sample_rate = librosa.load(kaggle_file_path , res_type='kaiser_best' )   

                # use a try/except to handle differences in kwargs between
                # featureset functions
                featureset_feature_matrix = np.array([[]])
                try:
                    featureset_feature_matrix = featureset_function(y=audio_data, sr=sample_rate)
                except TypeError:
                    featureset_feature_matrix = featureset_function(y=audio_data)

                # trim columns as necessary and append flattened matrix row-wise
                if feature_kaggle_matrix.size == 0:
                    #prasanth
                    print(featureset_feature_matrix.shape)
                    feature_mean_kaggle_matrix = np.append(np.mean(featureset_feature_matrix , axis = 1 , keepdims = True  ) , np.var(featureset_feature_matrix , axis = 1 , keepdims = True) , axis = 1)
                    feature_mean_kaggle_matrix = np.append(feature_mean_kaggle_matrix , np.max(featureset_feature_matrix , axis = 1 , keepdims = True ) , axis = 1)
                    feature_mean_kaggle_matrix = np.append(feature_mean_kaggle_matrix , np.min(featureset_feature_matrix, axis = 1 , keepdims = True) , axis = 1 )
                    feature_mean_kaggle_matrix = np.squeeze(feature_mean_kaggle_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T
                    feature_kaggle_matrix = np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T
                else:


                    #prasanth

                    dummy = np.append(np.mean(featureset_feature_matrix , axis = 1, keepdims = True) , 
                                            np.append( np.var(featureset_feature_matrix , axis = 1, keepdims = True) ,
                                                                np.append( np.max(featureset_feature_matrix , axis = 1, keepdims = True) , 
                                                                                            np.min(featureset_feature_matrix, axis = 1, keepdims = True) , axis = 1 ) , axis = 1), axis = 1)
                    dummy = np.squeeze(dummy.flatten('F').reshape(1, -1)).reshape(-1,1).T
                    feature_mean_kaggle_matrix = np.append(feature_mean_kaggle_matrix , dummy , axis = 0)
                    # feature_mean_kaggle_matrix = np.append(feature_mean_kaggle_matrix ,
                                                                #    np.squeeze(np.append(np.mean(featureset_feature_matrix , axis = 1  ) , 
                                                                                                    # np.var(featureset_feature_matrix , axis = 1)).flatten('F').reshape(1, -1)).reshape(-1,1).T , axis =0)
                    #prasanth




                    max_cols = min(featureset_feature_matrix.size, feature_kaggle_matrix.shape[1])
                    feature_kaggle_matrix = np.append(
                        feature_kaggle_matrix[:, :max_cols],
                        np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T[:, :max_cols],
                        axis=0
                    )

        # at this point, feature_training_matrix and feature_kaggle_matrix should hold the unprocessed feature data


        # training and kaggle matrices must have the same number of features for PCA to work correctly
        max_cols = min(feature_training_matrix.shape[1], feature_kaggle_matrix.shape[1])
        feature_training_matrix = feature_training_matrix[:, :max_cols]
        feature_kaggle_matrix = feature_kaggle_matrix[:, :max_cols]

        # perform train/test split on feature_training_matrix
        training_matrix_labels = np.array(training_matrix_labels)
        X_training = feature_training_matrix[~np.isin(np.arange(feature_training_matrix.shape[0]), test_sample_indices)]
        X_testing = feature_training_matrix[test_sample_indices]

        mean_training = feature_mean_train_matrix[~np.isin(np.arange(feature_training_matrix.shape[0]), test_sample_indices)]
        mean_testing = feature_mean_train_matrix[test_sample_indices]

        #perform train/test split on feature_mean_train_matrix
        print(feature_mean_train_matrix.shape)
        # fit PCA and SC objects on train split
        sc = StandardScaler()
        pca = PCA(n_components=0.95)
        X_training = sc.fit_transform(X_training)
        X_training = pca.fit_transform(X_training)
        

        # transform test split and kaggle data on PCA and SC objects from above
        X_testing = sc.transform(X_testing)
        X_testing = pca.transform(X_testing)
        feature_kaggle_matrix = sc.transform(feature_kaggle_matrix)
        feature_kaggle_matrix = pca.transform(feature_kaggle_matrix)

        sc = StandardScaler()
        pca = PCA(n_components=0.99)
        
        mean_training = pca.fit_transform(sc.fit_transform(mean_training))
        mean_testing = pca.transform(sc.transform(mean_testing))
        mean_kaggle  = pca.transform(sc.transform(feature_mean_kaggle_matrix))


        # append this feature matrix columnwise to the combined matrix
        if training_matrix_train.size == 0:
            mean_variance_train = mean_training
            mean_variance_test= mean_testing
            mean_variance_kaggle = mean_kaggle

            training_matrix_train = X_training
            training_matrix_test = X_testing
            kaggle_matrix = feature_kaggle_matrix
        else:
            mean_variance_train = np.append(mean_variance_train , mean_training , axis = 1)
            mean_variance_test = np.append(mean_variance_test , mean_testing , axis = 1)
            mean_variance_kaggle = np.append(mean_variance_kaggle, mean_kaggle , axis = 1)

            training_matrix_train = np.append(training_matrix_train, X_training, axis=1)
            training_matrix_test = np.append(training_matrix_test, X_testing, axis=1)
            kaggle_matrix = np.append(kaggle_matrix, feature_kaggle_matrix, axis=1)
    
    y_training = training_matrix_labels[~np.isin(np.arange(len(training_matrix_labels)), test_sample_indices)]
    y_testing = training_matrix_labels[test_sample_indices]

    # write resulting matrices to file
    np.save('X_train', mean_variance_train)
    np.save('X_test', mean_variance_test)
    np.save('X_kaggle', mean_variance_kaggle)
    np.save('y_train', y_training)
    np.save('y_test', y_testing)

    print(mean_variance_train.shape , mean_variance_test.shape , mean_variance_kaggle.shape)
    return (mean_variance_train, mean_variance_test, y_training, y_testing, mean_variance_kaggle)


def generate_one_hot(Y_training: np.array ) -> np.array:
    """
    generate a one hot encoded 2d matrix for all the given targets in the training data set 

    source : StackOverFlow

    url : https://stackoverflow.com/questions/58676588/how-do-i-one-hot-encode-an-array-of-strings-with-numpy 
    """
    onehotencoder = preprocessing.OneHotEncoder(categories='auto', sparse_output=False)
    one_hot_array = onehotencoder.fit_transform( Y_training)
    return one_hot_array, onehotencoder.categories_[0]
    

def standardize_columns(df: pd.DataFrame ) -> np.array:
    scaler = preprocessing.RobustScaler()
    df = scaler.fit_transform(df)
    return (df, scaler)

def normalize_row(array : np.array) -> np.array :
    return preprocessing.normalize(array , axis = 1)
def normalize_columns(array : np.array ) -> np.array :
    return preprocessing.normalize(array , axis = 0)


if __name__ == '__main__':
    """ Main function for testing
    """