# random forest imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

# SVM imports
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# gaussian naive bayes imports
from sklearn.naive_bayes import GaussianNB

# gradient boosting imports
from sklearn.ensemble import GradientBoostingClassifier

from utils.process_audio_data import combine_files_save_to_one
from utils.consts import training

def train_library_random_forest():

    print('Starting random forest classifier')

    # get training data
    training_df = combine_files_save_to_one(training)
    Y = training_df[training_df.columns[-1]]
    X = training_df.drop(training_df.columns[-1], axis=1)

    # perform train test split
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, stratify=Y)

    # create classifier model for training data
    print('Creating classifier...')
    classifier = RandomForestClassifier(max_depth=5)
    classifier.fit(X_train, Y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(Y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


def train_library_svm():

    print('Starting SVM classifier')

    # get training data
    training_df = combine_files_save_to_one(training)
    Y = training_df[training_df.columns[-1]]
    X = training_df.drop(training_df.columns[-1], axis=1)

    # perform train test split
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, stratify=Y)

    # create classifier model for training data
    print('Creating classifier...')
    classifier = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    classifier.fit(X_train, Y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(Y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


def train_library_naive_bayes():

    print('Starting naive bayes classifier')

    # get training data
    training_df = combine_files_save_to_one(training)
    Y = training_df[training_df.columns[-1]]
    X = training_df.drop(training_df.columns[-1], axis=1)

    # perform train test split
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, stratify=Y)

    # create classifier model for training data
    print('Creating classifier...')
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(Y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


def train_library_gradient_boosting():

    print('Starting gradient boosting classifier')

    # get training data
    training_df = combine_files_save_to_one(training)
    Y = training_df[training_df.columns[-1]]
    X = training_df.drop(training_df.columns[-1], axis=1)

    # perform train test split
    X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, stratify=Y)

    # create classifier model for training data
    print('Creating classifier...')
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    classifier.fit(X_train, Y_train)

    # validate model on testing data
    print('Evaluating classifier...')
    acc = balanced_accuracy_score(Y_test, classifier.predict(X_test))
    print(f'Balanced accuracy is {acc}')


if __name__== '__main__':
    train_library_random_forest()