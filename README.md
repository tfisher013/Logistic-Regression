# Logistic-Regression

## Description

UNM CS 529 Project 2: Creation of logistic regression classifier from scratch.

## Instructions for Use

### Install Dependencies
```bash
- python -m venv YOURVENV
- YOURENV/Scripts/activate
- pip install requirements.txt
```

### Train Logistic Regression

1. Specify values for the variables below in `utils/consts.py`:
   - `training_data_path`: the path to your training data directory
   - `testing_data_path`: the path to your testing (kaggle) data directory

2. Run `python -m training.train_logistic_regression` from the top level directory.

3. The following files will be generated:
- A file will be generated containing the trained model and saved in the `models/` directory.
- A file will be generated containing kaggle predictions and saved as `kaggle_predictions.csv` in the top level directory.


### Code Manifest
| File Name | Description |
| --- | --- |
| `training/train_logistic_regression.py` | This file contains the implementation of logistic regression and gradient descent. |
| `training/library_models.py` | This file contains our implementation of training and validation of scikit-learn ML models for comparison to our logistic regression implementation.  |
| `plots/convergence_plots.py` | This file contains a script for generating convergence rate plots.  |
| `utils/process_audio_data.py` | This file contains our feature extraction and transformation implementation.  |
| `utils/consts.py` | This file has constants used throughout the library.  |
| `utils/file_utils.py` | This file contains utility functions for working with files. |
| `validation/validate.py` | This file contains our function to generate kaggle prediction CSV files. |


## Developer Contributions

Prasanth Guvvala
- Implemented gradient descent algorithm.
- Implemented statistical feature transformations.
- Wrote script to generate confusion matrix.

Thomas Fisher
- Implemented combined feature extraction algorithm.
- Implemented `scikit-learn` library functions.
- Wrote script to plot convergence rate.

## kaggle Submission

Leaderboard position 5 achieved with accuracy 0.71 on April 8th.

