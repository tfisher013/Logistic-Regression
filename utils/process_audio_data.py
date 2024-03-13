import os
import librosa
import numpy as np
from sklearn.decomposition import PCA

feature_file_dir = 'feature_files'

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

    for audio_filename in os.listdir(audio_file_dir):

        if audio_filename == '.DS_Store':
            continue

        audio_filename_no_extension = audio_filename.replace('.au', '')
        csv_output_file_path = os.path.join(feature_file_dir, 'mfcc-' + audio_filename_no_extension + '.csv')

        audio_file_path = os.path.join(audio_file_dir, audio_filename)
        if os.path.isfile(audio_file_path):

            print(audio_file_path)

            audio_data, sample_rate = librosa.load(audio_file_path)

            # add featues per parameter selection
            if use_mfcc:
                mcff = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
                mcff_pca = pca.fit_transform(mcff.T)
                with open(csv_output_file_path, 'w') as f:
                    np.savetxt(f, mcff_pca.flatten('F').reshape(1, -1), delimiter=",")
                #print(mcff_pca)

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

    get_audio_features('../data/train/blues')
