# main.py

from Library.Import import *
from Library.Utilities import read_XY
from Library.Build_Reservoir import RC_run, RC_write_multiple
import time
import numpy as np

# Configuration and Input Section
DIRECTORY = './'
run = 'selective-obj'
# Hyperparameters and settings 
seed = 1
np.random.seed(seed=seed)
xfold = 1
repeat = 1
precision = 0
train_rate = 1.0e-4
n_hidden_prior = -1 # -1 binary input,  >0 ANN 
hidden_dim_prior = 0
activation_prior='' # '' or 'sharp_sigmoid' or 'relu'
n_hidden_post = -1 #  -1 a scaler is applied, >0 a ANN is used
hidden_dim_post = 0
temperature = False
multiple = -1 # -1 no stats > 0 nbr of reservoirs to get stats 
weight_pred_true_media = 0 # Loss to collect already generated media
failure = 10
data_link = 'test/data/and'

if run == 'generative-obj': 
    mode = 'AMN_objective'
    epochs = 100
    n_hidden_prior = 1
    hidden_dim_prior = 28 
    activation_prior= 'sharp_sigmoid' 
    weight_pred_true_media = 0.5

if run == 'selective-obj': # For iML1515EXP only
    mode = 'AMN_objective'
    epochs = 100
    n_hidden_prior = 3 
    hidden_dim_prior = 280
    activation_prior='gumbel_softmax' 
    weight_pred_true_media = 1 
    temperature = True

# Data loading function
def load_data(dataset_path):
    H, X, Y = read_XY(dataset_path, nY=1, scaling='XY')
    return X, Y

# Training function
def train_model(X_train, Y_train, reservoir_file):
    start_time = time.time()
    
    model, _, _, _, _, _, _ = RC_run(
        reservoir_file, X_train, Y_train,
        mode=mode,
        n_hidden_prior=n_hidden_prior,
        n_hidden_post=n_hidden_post,
        hidden_dim_prior=hidden_dim_prior,
        hidden_dim_post=hidden_dim_post,
        activation_prior=activation_prior,
        train_rate=train_rate,
        precision=precision,
        temperature=temperature,
        failure=failure,
        weight_pred_true_media=weight_pred_true_media,
        repeat=repeat,
        xfold=xfold,
        epochs=epochs,
        verbose=False
    )
    
    delta_time = time.time() - start_time
    print(f'Training complete. CPU time: {delta_time:.2f} seconds')
    
    return model

# Testing function
def test_model(model, X_test):
    # Use the model to predict on test data
    pred = model.predict(X_test)
    return pred

# Main execution
if __name__ == "__main__":
    s = 'iML1515EXP'
    reservoirname = f'{s}_train_AMN_QP'
    reservoirfile = f'{DIRECTORY}Reservoir/{reservoirname}'

    # Load your dataset (modify the path to your specific dataset)
    dataset_path = f'{DIRECTORY}{data_link}'
    X, Y = load_data(dataset_path)

    # Split the data into training and testing (implement your own split logic as needed)
    split_ratio = 0.8
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    # Train the model
    model = train_model(X_train, Y_train, reservoirfile)

    # Test the model
    y_test_pred = test_model(model, X_test)
    print("Test predictions:", y_test_pred)

