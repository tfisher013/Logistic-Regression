from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# the percentage error to allow before terminating
# gradient descent
epsilon = 0.001

# step size used for gradient descent
eta = 0.01

# regularization penalty
lambda_hyperparameter = 0.001

# number of features extracted from audio files
num_features = 40

# the max number of iterations of gradient descent before
# it is stopped
max_iterations = 50000

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

feature_file_dir_test = 'feature_files_test'

#making standardization global is good for transforming the real testing prediction data
sc = StandardScaler()

#making pca global is good for same reasons
pca = PCA(n_components=0.7)

# dictionary to map PCA objects to featuresets
feature_extraction_method_dict = dict()