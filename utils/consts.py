# the percentage error to allow before terminating
# gradient descent
epsilon = 0.001

# step size used for gradient descent
eta = 0.001

# regularization penalty
lambda_hyperparameter = 0.0001

# number of features extracted from audio files
num_features = 40

# the directory used to hold the feature CSV file
feature_file_dir = 'feature_files'

# the directory used to hold trained model files
model_dir = 'models'

training = 'train'

testing = 'test'

columns = list(range(0 , num_features+2))

model = 'model.csv'

others = 'others'

true = 'true'