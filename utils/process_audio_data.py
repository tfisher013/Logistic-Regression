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
from sklearn.preprocessing import StandardScaler


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

            audio_data, sample_rate = librosa.load(training_file_path)

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
    scaler = preprocessing.StandardScaler()
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