import csv
import os
import librosa
import numpy as np
from sklearn.decomposition import PCA
from utils.consts import feature_file_dir,num_features,training,columns
import pandas as pd
from sklearn import preprocessing 


def generate_one_hot(Y_training: np.array ) -> np.array:
    """
    generate a one hot encoded 2d matrix for all the given targets in the training data set 

    source : StackOverFlow

    url : https://stackoverflow.com/questions/58676588/how-do-i-one-hot-encode-an-array-of-strings-with-numpy 
    """
    onehotencoder = preprocessing.OneHotEncoder(categories='auto', sparse_output=False)
    print(Y_training.to_numpy()[:, np.newaxis])
    one_hot_array = onehotencoder.fit_transform( Y_training.to_numpy()[:, np.newaxis])
    return one_hot_array, onehotencoder.categories_[0]
    

def standardize_columns(df: pd.DataFrame ) -> np.array:
    scaler = preprocessing.StandardScaler()
    df = scaler.fit_transform(df)
    return df

def normalize_columns(df: pd.DataFrame) -> np.array :
    return preprocessing.normalize(df)

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

    for training_file in os.listdir(target_path):
        training_file_path = os.path.join(target_path, training_file)
        if os.path.isfile(training_file_path) and training_file.startswith('.') == False:

            audio_data, sample_rate = librosa.load(training_file_path)

            # extract and write mcff features
            mcff = librosa.feature.mfcc(y=audio_data, sr=sample_rate)

            # record the standard mcff shape so all other files also
            # share these dimensions
            if first_file:
                max_rows = mcff.shape[0]
                max_cols = mcff.shape[1]
                first_file = False

            max_cols = 1200

            write_matrix_to_csv(csv_output_file_path, mcff[:, :max_cols])
            #print(f'  file {training_file} written to with shape { mcff[:max_rows, :max_cols].shape}')


    return csv_output_file_path




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