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
                max_cols = 25
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
        feature_extraction_method_df_list[0] = perform_PCA_testing(feature_extraction_method_df_list[0], 'mfcc')
        feature_extraction_method_df_list[1] = perform_PCA_testing(librosa.amplitude_to_db(feature_extraction_method_df_list[1]), 'stft')

    combined_feature_data = np.append(feature_extraction_method_df_list[0] , feature_extraction_method_df_list[1], axis=1)
    
    return combined_feature_data , file_names



def create_test_train_split(training_data_directory: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    """

    # remove any existing feature files
    if os.path.exists(feature_file_dir):
        shutil.rmtree(feature_file_dir)

    # determine the number of classes
    class_labels = []
    for class_dir in os.listdir(training_data_directory):
        if os.path.isdir(class_dir):
            class_labels.append(class_dir)

    # create dataframes to hold the combined train/test data
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame() 

    print(f'the contents of {training_data_directory} are {os.listdir(training_data_directory)}')
    for class_dir in os.listdir(training_data_directory):
        class_dir_path = os.path.join(training_data_directory, class_dir)
        if os.path.isdir(class_dir_path) and class_dir.startswith('.') == False:

            # convert feature data to CSV and DataFrame formats
            feature_data_file = generate_target_csv(class_dir_path)
            feature_df = pd.read_csv(feature_data_file, header=None)

            print(feature_df)

            # perform train test split on feature data
            feature_X_train, feature_X_test, feature_y_train, feature_y_test = train_test_split(
                feature_df.drop(feature_df.columns[-1], axis=1), 
                feature_df[feature_df.columns[-1]], 
                test_size=0.01)
            
            # append feature train/test data to entire train/test data
            X_train = pd.concat([X_train, feature_X_train], axis=0, ignore_index=True)
            X_test = pd.concat([X_test, feature_X_test], axis=0, ignore_index=True)
            y_train = pd.concat([y_train, feature_y_train], axis=0, ignore_index=True)
            y_test = pd.concat([y_test, feature_y_test], axis=0, ignore_index=True)

    print(f'1. dimension of X_train before standardization and PCA: {X_train.shape}')
    print(X_train)

    # standardize columns with distributions generated
    # by training set
    X_train = pd.DataFrame(sc.fit_transform(X_train))
    X_test = pd.DataFrame(sc.transform(X_test))

    print(f'2. dimension of X_train after standardization: {X_train.shape}')
    print(X_train)
            
    # perform PCA using features selected from training
    # set
    X_train = pd.DataFrame(pca.fit_transform(X_train))
    X_test = pd.DataFrame(pca.transform(X_test))

    # normalize data columnwise
    X_train = pd.DataFrame(normalize_columns(X_train))
    X_test = pd.DataFrame(normalize_columns(X_test))

            
    print(f'3. dimension of X_train after standardization and PCA: {X_train.shape}')
    print(X_train)
    print(y_train)

    # add column of ones to X_train and X_test
    X_train[len(X_train.columns)] = 1
    X_test[len(X_test.columns)] = 1

    return (X_train, X_test, y_train, y_test)

def generate_one_hot(Y_training: np.array ) -> np.array:
    """
    generate a one hot encoded 2d matrix for all the given targets in the training data set 

    source : StackOverFlow

    url : https://stackoverflow.com/questions/58676588/how-do-i-one-hot-encode-an-array-of-strings-with-numpy 
    """
    onehotencoder = preprocessing.OneHotEncoder(categories='auto', sparse_output=False)
    # print(Y_training)
    one_hot_array = onehotencoder.fit_transform( Y_training)
    return one_hot_array, onehotencoder.categories_[0]
    

def standardize_columns(df: pd.DataFrame ) -> np.array:
    scaler = preprocessing.StandardScaler()
    df = scaler.fit_transform(df)
    return (df, scaler)

def normalize_columns(array : np.array) -> np.array :
    return preprocessing.normalize(array , axis = 1)

def combine_files_save_to_one(mode: str) -> pd.DataFrame:
    files = os.listdir(feature_file_dir)
    df = pd.DataFrame()
    for file in files:
        df = pd.concat([df, pd.read_csv(os.path.join(feature_file_dir, file), 
                                        header=None , names=columns, index_col= False)])
    df_new = pd.DataFrame(normalize_columns(
        df.drop(columns=[columns[-2], columns[-1]], axis=1)), columns=columns[:-2])
    df_new[columns[-1]] = list(df[columns[-1]]) 
    df_new[columns[-2]] = 1
    df = df_new
    df.to_csv(os.path.join(feature_file_dir, f'{mode}.csv'), index_label=False)
    return df


def write_matrix_to_csv(file_path: str, mat: np.array):
    """ Writes the provided numpy array as a row in the provided CSV file by flattening

        Parameters:
            file_path: the path to the csv file to which the row should be written
            mat: a numpy array
    """

    with open(file_path, 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(np.append(np.squeeze(mat.flatten('F').reshape(1, -1)), 
                                 [file_path.split("\\")[-1].split('-')[0]]))
        
        # print(f'  Writing row of length {len(np.append(np.squeeze(mat.flatten('F').reshape(1, -1)), 
        #                          [1, file_path.split("\\")[-1].split('-')[0]]))}')


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

                # iterate over each file in the class
                for training_file in os.listdir(class_dir_path):
                    training_file_path = os.path.join(class_dir_path, training_file)
                    if os.path.isfile(training_file_path) and training_file.startswith('.') == False:

                        audio_data, sample_rate = librosa.load(training_file_path)

                        # extract and write mcff features
                        mcff = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
                        stft = librosa.stft(audio_data)

                        # record the standard mcff shape so all other files also
                        # share these dimensions
                        # print(mcff.shape)
                        if first_file:
                            max_rows = mcff.shape[0]
                            max_cols = mcff.shape[1]
                            first_file = False
                            max_cols = 25
                            feature_extraction_method_df_list[0] = np.squeeze(mcff[:, :max_cols].flatten('F').reshape(1, -1)).reshape(-1,1).T
                            feature_extraction_method_df_list[1] = np.squeeze(stft[: , : max_cols ].flatten('F').reshape(1, -1)).reshape(-1,1).T
                        else:

                        
                        # print(np.squeeze(mcff[:, :max_cols].flatten('F').reshape(1, -1)).shape)
                        
                        #write_matrix_to_csv(os.path.join(feature_file_dir, class_label + '-mfcc-features.csv'), mcff[:, :max_cols]])
                            feature_extraction_method_df_list[0] = np.append(feature_extraction_method_df_list[0], np.squeeze(mcff[:, :max_cols].flatten('F').reshape(1, -1)).reshape(-1,1).T , axis = 0) 
                            # print(feature_extraction_method_df_list[0].shape)
                            # print(targets.shape)                        
                        # print(feature_extraction_method_df_list[0])

                        # stft features
                        
                            feature_extraction_method_df_list[1] = np.append(feature_extraction_method_df_list[1], np.squeeze(stft[: , : max_cols ].flatten('F').reshape(1, -1)).reshape(-1,1).T , axis = 0 )
                        # print(feature_extraction_method_df_list[1].shape)
                        targets = np.append(targets , training_file_path.split("\\")[-1].split('-')[0].split('.')[0])
    # now that all samples have been written to file, process data
    feature_extraction_method_df_list[0] = perform_PCA_training(feature_extraction_method_df_list[0], 'mfcc')
    feature_extraction_method_df_list[1] = perform_PCA_training(librosa.amplitude_to_db(feature_extraction_method_df_list[1]), 'stft')
    targets = targets.reshape(-1,1)
    # print(targets.shape)

    combined_feature_data = np.append(feature_extraction_method_df_list[0] , feature_extraction_method_df_list[1], axis=1)
    
    return combined_feature_data , targets




def get_audio_features(audio_file_dir: str,
                       use_mfcc: bool=True,
                       use_chroma: bool=False,
                       use_stft: bool=False,
                       use_spectral: bool=False,
                       use_zcr: bool=False):
    """
    """

    pca = PCA(n_components=2)

    if not os.path.isdir(feature_file_dir):
        os.mkdir(feature_file_dir)

    csv_output_file_path = os.path.join(feature_file_dir, 'mfcc-features-' + os.path.basename(feature_file_dir) + '.csv')

    for (file_index, audio_filename) in enumerate(os.listdir(audio_file_dir)):

        if audio_filename == '.DS_Store':
            continue

        audio_filename_no_extension = audio_filename.replace('.au', '')

        audio_file_path = os.path.join(audio_file_dir, audio_filename)
        if os.path.isfile(audio_file_path):

            audio_data, sample_rate = librosa.load(audio_file_path)

            # add featues per parameter selection
            if use_mfcc:
                mcff = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
                mcff_pca = pca.fit_transform(mcff.T)

                feature_content = [audio_filename_no_extension]
                feature_content.extend(np.squeeze(mcff_pca.flatten('F').reshape(1, -1)).tolist())
                    
                with open(csv_output_file_path, 'a', newline='') as csvfile:

                    write = csv.writer(csvfile)

                    # write header row for first file
                    if file_index == 0:
                        write.writerow(['file'] + ['Coeff-' + str(i+1) for i in range(len(feature_content) - 1)])

                    write.writerow(feature_content)

            if use_stft:
                stft = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
                print(stft)

            if use_chroma:
                chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sample_rate)
                print(chroma)

            if use_spectral:
                spectral = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
                print(spectral)

            if use_zcr:
                zcr = librosa.feature.zero_crossing_rate(y=audio_data)
                print(zcr)

            # perform CPA on audio features

            # write features to file


if __name__ == '__main__':

    generate_target_csv('../data/train/blues')
    #get_audio_features('../data/train/blues')