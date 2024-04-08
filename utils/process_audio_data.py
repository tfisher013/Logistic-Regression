import csv
import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from utils.consts import feature_file_dir,num_features,training,columns , sc , pca , feature_extraction_method_dict
import pandas as pd
from sklearn import preprocessing 
from  typing import Tuple
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , RobustScaler
import random


def combined_data_processing(training_data_dir: str, testing_data_dir: str) -> tuple[np.array, np.array]:
    """ Processes both training and kaggle data at once. Splits training data into train and test splits.
        All train/test and kaggle data undergo standardization, PCA, and normalization.
    """

    # define the featureset functions
    """commented for now"""
    featureset_functions = [
                            # librosa.feature.zero_crossing_rate,
                            librosa.feature.mfcc, 
                            # librosa.feature.chroma_stft, 
                            # librosa.feature.chroma_cqt,
                            librosa.feature.spectral.spectral_contrast,
                            # librosa.feature.chroma_cens
                            ]

    # create empty arrays to hold training and kaggle datasets
    training_matrix_train = np.array([[]])
    training_matrix_test = np.array([[]])
    kaggle_matrix = np.array([[]])

    # define empty array to hold the labels for the training matrix
    training_matrix_labels = []

    # define which row indices will be for training vs. testing; this needs to be done
    # so the test train splits are for the same samples across all featuresets
    num_classes = 10
    num_samples_per_class = 90
    test_size = 0.1
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

        # populate feature training matrix
        for class_dir in os.listdir(training_data_dir):
            class_dir_path = os.path.join(training_data_dir, class_dir)
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
                            feature_training_matrix = np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T
                        else:

                            max_cols = min(featureset_feature_matrix.size, feature_training_matrix.shape[1])

                            feature_training_matrix = np.append(
                                feature_training_matrix[:, :max_cols],
                                np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T[:, :max_cols],
                                axis=0
                            )

                        if featureset_idx == 0:
                            training_matrix_labels.append(class_dir)

        # populate kaggle feature matrix
        for kaggle_file in os.listdir(testing_data_dir):
            kaggle_file_path = os.path.join(testing_data_dir, kaggle_file)
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
                    feature_kaggle_matrix = np.squeeze(featureset_feature_matrix.flatten('F').reshape(1, -1)).reshape(-1,1).T
                else:
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

        # fit PCA and SC objects on train split
        sc = StandardScaler()
        pca = PCA(n_components=0.95)
        X_training = sc.fit_transform(X_training)
        X_training = pca.fit_transform(X_training)
        

        # transform test split and kaggle data on PCA and SC objects from above
        X_testing = sc.transform(X_testing)
        #kaggle_matrix = sc.transform(kaggle_matrix)
        X_testing = pca.transform(X_testing)
        feature_kaggle_matrix = sc.transform(feature_kaggle_matrix)
        feature_kaggle_matrix = pca.transform(feature_kaggle_matrix)

        # normalize columns
        # X_training = preprocessing.normalize(X_training, axis = 0)
        # X_testing = preprocessing.normalize(X_testing, axis = 0)

        # append this feature matrix columnwise to the combined matrix
        if training_matrix_train.size == 0:
            training_matrix_train = X_training
            training_matrix_test = X_testing
            kaggle_matrix = feature_kaggle_matrix
        else:
            training_matrix_train = np.append(training_matrix_train, X_training, axis=1)
            training_matrix_test = np.append(training_matrix_test, X_testing, axis=1)
            kaggle_matrix = np.append(kaggle_matrix, feature_kaggle_matrix, axis=1)

    y_training = training_matrix_labels[~np.isin(np.arange(len(training_matrix_labels)), test_sample_indices)]
    y_testing = training_matrix_labels[test_sample_indices]

    # write resulting matrices to file
    #np.append(training_matrix_train, y_training.reshape(-1, 1), axis=1).tofile('training_data_with_labels.csv', sep=',')
    #np.append(training_matrix_test, y_testing.reshape(-1, 1), axis=1).tofile('testing_data_with_labels.csv', sep=',')
    #kaggle_matrix.tofile('kaggle_data_features.csv', sep=',')

    return (training_matrix_train, y_training, training_matrix_test, y_testing, kaggle_matrix)


def perform_PCA_training(feature_matrix: pd.DataFrame, feature_extraction_method_name: str) -> np.array:
    """
    """

    # feature_matrix.drop(feature_matrix.columns[-1])

    # first standardize data along columns
    standardized_matrix, sc = standardize_columns(feature_matrix)

    # perform PCA
    pca = PCA()
    standardized_matrix= pca.fit_transform(standardized_matrix)

    # normalize by column
    standardized_matrix = normalize_columns(standardized_matrix)

    # update dictionary with data processing objects
    feature_extraction_method_dict[feature_extraction_method_name] = [pca, sc]
    # print(standardized_matrix)

    return standardized_matrix

def perform_PCA_testing(feature_matrix: pd.DataFrame, feature_extraction_method_name: str) -> np.array:
    """
    """

    # first standardize data along columns
    sc = feature_extraction_method_dict[feature_extraction_method_name][1]
    standardized_matrix = sc.transform(feature_matrix)

    # perform PCA
    pca = feature_extraction_method_dict[feature_extraction_method_name][0]
    standardized_matrix = pca.transform(standardized_matrix)

    # normalize by column
    standardized_matrix = normalize_columns(standardized_matrix)

    return standardized_matrix


def process_test_data(test_data_directory: str) -> pd.DataFrame:
    file_names = []
    first_file = True
    feature_extraction_method_df_list = [pd.DataFrame() , pd.DataFrame()] 
    class_dir_path = test_data_directory
    # iterate over each file in the class
    for training_file in os.listdir(class_dir_path):
        file_names.append(training_file)
        training_file_path = os.path.join(class_dir_path, training_file)
        if os.path.isfile(training_file_path) and training_file.startswith('.') == False:

            audio_data, sample_rate = librosa.load(training_file_path , res_type='kaiser_fast')

            # extract and write mcff features
            mcff = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
            stft = librosa.stft(audio_data)

            # record the standard mcff shape so all other files also
            # share these dimensions
            if first_file:
                max_rows = mcff.shape[0]
                max_cols = mcff.shape[1]
                first_file = False
                max_cols = 1200
                feature_extraction_method_df_list[0] = np.squeeze(mcff[:, :max_cols].flatten('F').reshape(1, -1)).reshape(-1,1).T
                feature_extraction_method_df_list[1] = np.squeeze(stft[: , : max_cols ].flatten('F').reshape(1, -1)).reshape(-1,1).T
            else:

            
            # print(np.squeeze(mcff[:, :max_cols].flatten('F').reshape(1, -1)).shape)
            
            #write_matrix_to_csv(os.path.join(feature_file_dir, class_label + '-mfcc-features.csv'), mcff[:, :max_cols]])
                feature_extraction_method_df_list[0] = np.append(feature_extraction_method_df_list[0], np.squeeze(mcff[:, :max_cols].flatten('F').reshape(1, -1)).reshape(-1,1).T , axis = 0) 
                # print(feature_extraction_method_df_list[0].shape)                        
            # print(feature_extraction_method_df_list[0])

            # stft features
            
                feature_extraction_method_df_list[1] = np.append(feature_extraction_method_df_list[1], np.squeeze(stft[: , : max_cols ].flatten('F').reshape(1, -1)).reshape(-1,1).T , axis = 0 )
            # print(feature_extraction_method_df_list[1].shape)

# now that all samples have been written to file, process data
    #feature_extraction_method_df_list[0] = perform_PCA_testing(feature_extraction_method_df_list[0], 'mfcc')
    #feature_extraction_method_df_list[1] = perform_PCA_testing(librosa.amplitude_to_db(feature_extraction_method_df_list[1]), 'stft')
    feature_extraction_method_df_list[1] = librosa.amplitude_to_db(feature_extraction_method_df_list[1])

    combined_feature_data = np.append(feature_extraction_method_df_list[0] , feature_extraction_method_df_list[1], axis=1)
    
    return combined_feature_data,  file_names
    


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


def generate_target_csv(target_path: str) -> str:
    """ Generates a feature matrix from a directory of training instances

        Parameters:
            target_path: the path to the directory containing training
                instances all pertaining to a particular class

        Returns:
            the path to the generated csv file

    """
    training_data_directory = target_path

    class_label = os.path.basename(target_path)

    if not os.path.isdir(feature_file_dir):
        os.mkdir(feature_file_dir)
    csv_output_file_path = os.path.join(feature_file_dir, class_label + '-features.csv')

    # I'm not sure why, but one of the disco files gives an mcff result with
    # slightly more columns, even though the parameters are the same. Recording
    # the typical shape of the output to trim any outliers.
    max_rows = 0
    max_cols = 0
    first_file = True

    mfcc_df = np.array([[]])
    print(mfcc_df.shape)
    targets = np.array([])
    stft_df = np.array([[]])
    feature_extraction_method_df_list = [mfcc_df, stft_df]

    # iterate over each class directory
    for class_dir in os.listdir(training_data_directory):
            class_dir_path = os.path.join(training_data_directory, class_dir)
            if os.path.isdir(class_dir_path) and class_dir.startswith('.') == False:

                print('Beginning processing of class ' + class_dir)

                # iterate over each file in the class
                for training_file in os.listdir(class_dir_path):
                    training_file_path = os.path.join(class_dir_path, training_file)
                    if os.path.isfile(training_file_path) and training_file.startswith('.') == False:

                        audio_data, sample_rate = librosa.load(training_file_path)

                        # extract and write mcff features
                        mcff = librosa.feature.mfcc(y=audio_data, sr=sample_rate / 2)
                        #stft = librosa.stft(y=audio_data, n_fft=256, hop_length=1024)
                        #spectral = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)

                        # record the standard mcff shape so all other files also
                        # share these dimensions
                        # print(mcff.shape)
                        if first_file:
                            max_rows = mcff.shape[0]
                            max_cols = mcff.shape[1]
                            first_file = False
                            max_cols = 1200
                            feature_extraction_method_df_list[0] = np.squeeze(mcff.flatten('F').reshape(1, -1)).reshape(-1,1).T
                            #feature_extraction_method_df_list[1] = np.squeeze(stft.flatten('F').reshape(1, -1)).reshape(-1,1).T
                        else:
                        
                            max_cols_mfcc = min(feature_extraction_method_df_list[0].shape[1], mcff.shape[0] * mcff.shape[1])
                            #max_cols_stft = min(feature_extraction_method_df_list[1].shape[1], stft.shape[0] * stft.shape[1])

                            feature_extraction_method_df_list[0] = np.append(feature_extraction_method_df_list[0][:, :max_cols_mfcc], 
                                                                             np.squeeze(mcff.flatten('F').reshape(1, -1)).reshape(-1,1).T[:, :max_cols_mfcc], axis=0)
                            # feature_extraction_method_df_list[1] = np.append(feature_extraction_method_df_list[1][:, :max_cols_stft],
                            #                                                  np.squeeze(stft.flatten('F').reshape(1, -1)).reshape(-1,1).T[:, :max_cols_stft], axis=0)
                            
                        print(f'  mfcc.shape = ' + str(feature_extraction_method_df_list[0].shape))
                        #print(f'  stft.shape = ' + str(feature_extraction_method_df_list[1].shape))

                        targets = np.append(targets , training_file_path.split("\\")[-1].split('-')[0].split('.')[0])
    # now that all samples have been written to file, process data
    #feature_extraction_method_df_list[0] = perform_PCA_training(feature_extraction_method_df_list[0], 'mfcc')
    #feature_extraction_method_df_list[1] = perform_PCA_training(librosa.amplitude_to_db(feature_extraction_method_df_list[1]), 'stft')
    #feature_extraction_method_df_list[1] = librosa.amplitude_to_db(feature_extraction_method_df_list[1])
    targets = targets.reshape(-1,1)

    #combined_feature_data = np.append(feature_extraction_method_df_list[0], feature_extraction_method_df_list[1], axis=1)
    
    #return np.append(combined_feature_data, np.ones(combined_feature_data.shape[0]).reshape(-1, 1), 1) , targets
    return feature_extraction_method_df_list[0] , targets



if __name__ == '__main__':

    generate_target_csv('../data/train/blues')
    #get_audio_features('../data/train/blues')
