##############################################################################
# This library provide utilities for buiding, training, evaluating, saving
# and loading models. The actual model is passed through the parameter
# 'model_type'. The library makes use of Keras, tensorfow and sklearn
# The provided models are:
# - ANN_dense: a simple Dense neural network
# - AMN_QP: an ANN_dense with custom lost ala PINN
# Authors: Jean-loup Faulon, jfaulon@gmail.com 
# Updates: 24/11/2023, 12/04/2024, 22/06/2024 (KOs), 
#          28/10/2024 (regression replaced by scoring_function)
##############################################################################

from Library.Import import *

###############################################################################
# Custom functions for tf/keras 
###############################################################################

def sharp_sigmoid(x):
    # Custom activation function
    return K.sigmoid(10000.0 * x)
from tensorflow.keras.utils import get_custom_objects, CustomObjectScope
from tensorflow.keras.layers import Activation, Lambda
get_custom_objects().update({'sharp_sigmoid': Activation(sharp_sigmoid)})

def my_mse(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])

def my_mae(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    return keras.losses.mean_squared_error(y_true[:,:end], y_pred[:,:end])

def my_r2(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    yt, yp = y_true[:,:end], y_pred[:,:end]
    SS =  K.sum(K.square( yt-yp ))
    ST = K.sum(K.square( yt - K.mean(yt) ) )
    return 1 - SS/(ST + K.epsilon())

def my_binary_crossentropy(y_true, y_pred):
    # Custom loss function
    end = y_true.shape[1]
    return keras.losses.binary_crossentropy(y_true[:,:end], y_pred[:,:end])

def my_acc(y_true, y_pred):
    # Custom metric function
    end = y_true.shape[1]
    return keras.metrics.binary_accuracy(y_true[:,:end], y_pred[:,:end])
    
def CROP(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call x = crop(2,5,10)(x) to slice the second dimension
    from tensorflow.keras.layers import Lambda
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]
    return Lambda(func)

###############################################################################
# Custom Loss functions to evaluate models and compute gradients
# Inputs:
# - V: the (predicted) flux vector
# - Pout: the matrix selecting in V measured outgoing fluxes
# - Vout: the measured outgoing fluxes
# - Pin: the matrix selecting in V measured incoming fluxes
# - Pko: the matrix selecting in V ko fluxes
# - Vinko: the input fluxes
# - S: the stoichiometric matrix
# Outputs:
# - Loss and gradient
# ##############################################################################

def Loss_Vout(V, Vout, parameter):
    # Loss = ||Pout.V-Vout||
    # When Vout is empty just compute Pout.V
    # dLoss = ∂([Pout.V-Vout]^2)/∂V = Pout^T (Pout.V - Vout)
    Pout = tf.convert_to_tensor(np.float32(parameter.Pout))
    Loss = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True) - Vout
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pout.shape[0] # rescaled
    return Loss_norm

def Loss_SV(V, parameter):
    # Loss = ||SV||
    # dLoss =  ∂([SV]^2)/∂V = S^T SV
    S  = tf.convert_to_tensor(np.float32(parameter.S))
    Loss = tf.linalg.matmul(V, tf.transpose(S), b_is_sparse=True) 
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/S.shape[0] # rescaled
    return Loss_norm

def Loss_Vin(V, Vin, parameter):
    # Loss = ReLU(Pin . V - Vin)
    # dLoss = ∂(ReLU(Pin . V - Vin)^2/∂V
    Pin  = tf.convert_to_tensor(np.float32(parameter.Pin))
    Loss = tf.linalg.matmul(V, tf.transpose(Pin), b_is_sparse=True) - Vin 
    Loss = tf.keras.activations.relu(Loss)
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pin.shape[0] # rescaled
    return Loss_norm

def Loss_Vko(V, Vko, parameter):
    # Loss = Pko . V (for element at zero in Vko)
    # dLoss = ∂(Pko . V - Vko)^2/∂V  
    if parameter.Pko.shape[0] == 0: # zero Loss
        return tf.norm(0 * V, axis=1, keepdims=True)
    Pko  = tf.convert_to_tensor(np.float32(parameter.Pko))
    PkoV = tf.linalg.matmul(V, tf.transpose(Pko), b_is_sparse=True)
    Loss = tf.where(tf.equal(Vko, 1), PkoV, tf.zeros_like(PkoV)) # GPT4o  
    Loss_norm = tf.norm(Loss, axis=1, keepdims=True)/Pko.shape[0] # rescaled
    return Loss_norm

def Loss_constraint(V, Vinko, parameter):
    # mean squared sum L2+L3+L4+L5
    Lin, Lko = parameter.Pin.shape[0], parameter.Pko.shape[0]
    Vin = CROP(1, 0, Lin)(Vinko)  
    Vko = CROP(1, Lin, Lin+Lko)(Vinko)
    L2 = Loss_SV(V, parameter)
    L3 = Loss_Vin(V, Vin, parameter)
    L4 = Loss_Vko(V, Vko, parameter)
    # square sum of L2, L3, L4, L5
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    L = tf.math.reduce_sum(tf.concat([L2, L3, L4], axis=1), axis=1)
    # divide by number_constraint()
    L = tf.math.divide_no_nan(L, tf.constant(parameter.number_constraint, 
                                             dtype=tf.float32))
    return L

def Loss_all(V, Vinko, Vout, parameter):
    # mean square sum of L1, L2, L3, L4, L5
    if Vout.shape[0] < 1: # No target provided = no Loss_Vout
        return Loss_constraint(V, Vinko, parameter)

    Lin, Lko = parameter.Pin.shape[0], parameter.Pko.shape[0]
    Vin = CROP(1, 0, Lin)(Vinko)  
    Vko = CROP(1, Lin, Lin+Lko)(Vinko)
    L1 = Loss_Vout(V, Vout, parameter)
    L2 = Loss_SV(V, parameter)
    L3 = Loss_Vin(V, Vin, parameter)
    L4 = Loss_Vko(V, Vko, parameter)
    # square sum of L1, L2, L3, L4, L5
    L1 = tf.math.square(L1)
    L2 = tf.math.square(L2)
    L3 = tf.math.square(L3)
    L4 = tf.math.square(L4)
    L = tf.math.reduce_sum(tf.concat([L1, L2, L3, L4], axis=1), axis=1)
    # divide by number_constraint()
    L = tf.math.divide_no_nan(L, tf.constant(parameter.number_constraint+1, 
                                             dtype=tf.float32))
    return L

###############################################################################
# Dense model
# ##############################################################################

from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Reshape, multiply
from tensorflow.keras.layers import concatenate, add, subtract, dot

def input_ANN_Dense(parameter, verbose=False):
    # Shape X and Y depending on the model used
    from Library.Utilities import MaxScaler
    if parameter.scaler != 0: # Normalize X
        parameter.X, parameter.scaler = MaxScaler(parameter.X) 
    if verbose:
        print(f'ANN Dense scaler {parameter.scaler}')
    return parameter.X, parameter.Y
    
def Dense_layers(inputs, parameter, trainable=True, verbose=False):
    # Build a dense architecture with some hidden layers
    from tensorflow.keras.regularizers import l2
    
    activation=parameter.activation
    n_hidden=parameter.n_hidden
    dropout=parameter.dropout
    hidden_dim=parameter.hidden_dim
    output_dim=parameter.output_dim
    hidden = inputs
    n_hidden = 0 if hidden_dim == 0 else n_hidden
    activation_hidden = 'linear' if 'sigmoid' in activation else 'relu'
    for i in range(n_hidden):
        hidden = Dense(hidden_dim,
                       kernel_initializer='random_normal', #'glorot_uniform',  # 'random_normal',
                       bias_initializer='zeros',
                       activation=activation_hidden, 
                       trainable=bool(trainable)) (hidden)
        hidden = Dropout(dropout)(hidden)
    outputs = Dense(output_dim,
                    kernel_initializer='random_normal', #'glorot_uniform',  # 'random_normal',
                    bias_initializer='zeros',
                    kernel_regularizer=l2(0.01), # !!!
                    activation=activation, 
                    trainable=bool(trainable)) (hidden) 
    if verbose:
        print(f'Dense layer n_hidden: {n_hidden} hidden_dim: {hidden_dim} '\
              f'input_dim: {inputs.shape[1]} output_dim: {output_dim} '\
              f'activation: {activation} trainable: {trainable}')
        
    return outputs

def ANN_Dense(parameter, trainable=True, verbose=False):
    # A standard Dense model with several layers
    from sklearn.metrics import r2_score
    
    input_dim, output_dim = parameter.input_dim, parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs = Dense_layers(inputs, parameter,
                           trainable=trainable, verbose=verbose)
    model = keras.models.Model(inputs=[inputs], outputs=[outputs])
    regression = True if parameter.scoring_function == r2_score else False
    loss = 'mse' if regression else 'binary_crossentropy'
    metrics = ['mae'] if regression else ['acc']
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    if verbose: print(f'nbr parameters: {model.count_params()}')
    parameter.model = model

    return parameter

###############################################################################
# AMN_QP: a ANN_Dense trainable prior layer with mechanistic loss ala PINN
# ##############################################################################

def input_AMN(parameter, verbose=False):
    # Shape the IOs
    # IO: X and Y
    # For all
    # - add additional zero columns to Y
    #   the columns are used to minimize SV, Pin V ≤ Vin, Pko V
    from Library.Utilities import MaxScaler
    
    X, Y = parameter.X, parameter.Y
    if parameter.scaler != 0: # Normalize X
        X, parameter.scaler = MaxScaler(X) 
    if verbose: print(f'AMN scaler: {parameter.scaler}')
    y = np.zeros(Y.shape[0]).reshape(Y.shape[0],1)
    Y = np.concatenate((Y, y), axis=1) # SV constraint
    Y = np.concatenate((Y, y), axis=1) # Pin constraint
    Y = np.concatenate((Y, y), axis=1) # Pko constraint
    if 'QP' in parameter.model_type:
        if verbose: print(f'QP input shape: {X.shape} {Y.shape}',)
    else:
        print(parameter.model_type)
        sys.exit('This AMN type does not have input') 
    parameter.input_dim = parameter.X.shape[1]
    parameter.X, parameter.Y = X, Y
    
    return parameter.X, parameter.Y

def output_AMN(V, Vinko, parameter, verbose=False):
    # Get output for all AMN models
    # output = PoutV + constaints + V
    # where S and Pout are the stoichiometric and measurement matrix

    Lin, Lko = parameter.Pin.shape[0], parameter.Pko.shape[0]
    Vin = CROP(1, 0, Lin)(Vinko)  
    Vko = CROP(1, Lin, Lin+Lko)(Vinko)

    Pout  = tf.convert_to_tensor(np.float32(parameter.Pout))
    PoutV = tf.linalg.matmul(V, tf.transpose(Pout), b_is_sparse=True)
    SV    = Loss_SV(V, parameter) # SV const
    PinV  = Loss_Vin(V, Vin, parameter) # Pin const
    PkoV  = Loss_Vko(V, Vko, parameter) # Pko const
        
    # Return outputs = PoutV + SV + PinV + PkoV + V
    outputs = concatenate([PoutV, SV, PinV, PkoV, V], axis=1)
    parameter.output_dim = outputs.shape[1]
    if verbose:
        print(f'AMN output shapes for PoutV: {PoutV.shape} \
SV: {SV.shape} Pin: {PinV.shape} Pko: {PkoV.shape}  \
V: {V.shape} outputs: {outputs.shape}')

    return outputs
   
def QP_layers(inputs, parameter, trainable=True, verbose=False):
    # Build and return an architecture ala PINN 
    # An initial vector V is calculated via training
    # through a Dense layer
    # Inputs:
    # - input flux vector, 
    # - trainable: if False QP is run with pretrained weights
    #              defautl is true and weigth are calculated
    # Outputs:
    # - ouput_AMN (see function)

    param = copy.copy(parameter)
    param.output_dim = parameter.S.shape[1]
    param.activation = 'relu' # all fluxes positives
    V = Dense_layers(inputs, param, trainable=trainable, 
                     verbose=verbose)
    
    return output_AMN(V, inputs, parameter, verbose=verbose) 

def AMN_QP(parameter, trainable=True, verbose=False):
    # Build and return an AMN with training
    # input: problem parameter
    # output: Trainable model
    # Loss history is not recorded (already done through tf training)

    # Get dimensions and build model
    input_dim, output_dim = parameter.X.shape[1], parameter.output_dim
    inputs = Input(shape=(input_dim,))
    outputs = QP_layers(inputs, parameter, 
                        trainable=trainable, verbose=verbose)
    # Compile
    model = keras.models.Model(inputs=[inputs], outputs=outputs)
    loss, metrics = my_mse, [my_r2] 
    model.compile(loss=loss, optimizer='adam', metrics=metrics)
    if verbose == 2: print(model.summary())
    if verbose: print(f'nbr parameters: {model.count_params()}')
    parameter.model = model

    return parameter

################################################################################
# Train and Evaluate all models
# ##############################################################################

class ReturnStats:
    def __init__(self, v1, v2, v3, v4, v5, v6, v7, v8):
        self.train_objective = (v1, v2)
        self.train_loss = (v3, v4)
        self.test_objective = (v5, v6)
        self.test_loss = (v7, v8)
    def printout(self, jobname, time=0): 
        # Printing Stats
        print(f'Stats for {jobname} CPU-time {time:.4f}')
        print(f'{jobname} R2 = {self.train_objective[0]:.4f}' \
              f'± {self.train_objective[1]:.4f} '\
              f'Constraint = {self.train_loss[0]:.4f} ± {self.train_loss[1]:.4f}')
        print(f'{jobname} Q2 = {self.test_objective[0]:.4f} '\
              f'± {self.test_objective[1]:.4f} '\
              f'Constraint = {self.test_loss[0]:.4f} ± {self.test_loss[1]:.4f}')

def print_loss_evaluate(y_true, y_pred, Vinko, parameter):
    # Print all losses
    loss_out, loss_cst, loss_all = -1, -1, -1
    end = y_true.shape[1] - parameter.number_constraint
    nV = parameter.S.shape[1]
    V = y_pred[:, y_true.shape[1]:y_true.shape[1] + nV]
    Vout = y_true[:,:end]
    Lin, Lko = parameter.Pin.shape[0], parameter.Pko.shape[0]
    Vin = Vinko[:,:Lin]
    Vko = Vinko[:,Lin:Lin+Lko]
    L1 = Loss_Vout(V, Vout, parameter)
    L1 = np.mean(L1.numpy())
    L2 = Loss_SV(V, parameter)
    L2 = np.mean(L2.numpy())
    L3 = Loss_Vin(V, Vin, parameter)
    L3 = np.mean(L3.numpy())
    L4 = Loss_Vko(V, Vko, parameter)
    L4 = np.mean(L4.numpy())
    print(f'Loss Vout: {L1:.1E}')
    print(f'Loss SV:   {L2:.1E}')
    print(f'Loss Vin:  {L3:.1E}')
    print(f'Loss Vko:  {L4:.1E}')
    return

def get_loss_evaluate(x, y_true, y_pred, parameter, verbose=False):
    # Return loss on constraint for y_pred
    if 'AMN' in parameter.model_type:
        Vf = y_pred[:,y_true.shape[1]:y_true.shape[1]+parameter.S.shape[1]]
        Vinko = x
        if verbose:
            print_loss_evaluate(y_true, y_pred, Vinko, parameter)               
        loss = Loss_constraint(Vf, Vinko, parameter)
        loss = np.mean(loss.numpy())
    else:
        loss = -1    
    return loss

def evaluate_model(model, x, y_true, parameter, verbose=False):
    # Return y_pred, stats (R2/Acc) 
    # and error on constraints for regression and classification
    # if input model than x, y_true sent to input model
    from sklearn.metrics import r2_score
    
    y_pred = model.predict(x, verbose=verbose) # whole y prediction

    # Ignore constraints added to y_true
    end = y_true.shape[1] - parameter.number_constraint
    yt, yp = y_true[:,:end], y_pred[:,:end]
    if parameter.scoring_function == r2_score:
        obj = parameter.scoring_function(yt, yp, 
                                         multioutput='variance_weighted')
    else:
        obj = parameter.scoring_function(yt, yp.round()) 

    # compute stats on constraints
    loss = get_loss_evaluate(x, y_true, y_pred, parameter, verbose=verbose)
    stats  = ReturnStats(obj, 0, loss,  0, obj, 0, loss,  0)
 
    return y_pred, stats

def model_input(parameter, trainable=True, verbose=False):
    from Library.Build_Reservoir import input_RC
    # return input for the appropriate model_type
    if   'ANN_Dense' in parameter.model_type:
        return input_ANN_Dense(parameter, verbose=verbose)
    elif 'RC' in parameter.model_type:
        return input_RC(parameter, verbose=verbose)
    elif 'AMN_QP' in parameter.model_type:
        return input_AMN(parameter, verbose=verbose)    
    else:
        sys.exit(f'{parameter.model_type}: no input available')

def model_type(parameter, verbose=False):
    from Library.Build_Reservoir import RC
    # create the appropriate model_type
    if verbose:
        print('-----------------------------------', parameter.model_type)
    if 'ANN_Dense' in parameter.model_type:
        return ANN_Dense(parameter, verbose=verbose)
    elif 'RC' in parameter.model_type:
        return RC(parameter, verbose=verbose)
    elif 'AMN_QP' in parameter.model_type:
        return AMN_QP(parameter, verbose=verbose)
    else:
        print(parameter.model_type)
        sys.exit('not a trainable model')

class TemperatureAnnealingCallback(tf.keras.callbacks.Callback):
    # Temperature schedule on a log scale
    def __init__(self, initial_temperature, final_temperature, epochs):
        super(TemperatureAnnealingCallback, self).__init__()
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.decay = (np.log(final_temperature) - np.log(initial_temperature)) / (epochs - 1)
        self.epochs = epochs
    def on_epoch_begin(self, epoch, logs=None):
        global_training_temperature = self.initial_temperature * np.exp(self.decay * epoch)

def train_model(parameter, Xtrain, Ytrain, Xtest, Ytest, 
                temperature=0, failure=0, verbose=False):
    # A standard function to create a model, fit, and test
    # Inputs:
    # - all necessary parameters including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.scoring_function for regression or classification
    # Outputs:
    # - Net: the trained network
    # - ytrain, ytest: y values for training and test sets
    # - otrain, ltrain: objective and loss for training set
    # - otest, ltest: objective and loss for training set
    # - history: tf fit history
    # Must have verbose=2 to verbose the fit 

    def reinitialize_weights(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                for attr in ['gamma', 'beta', 'moving_mean', 'moving_variance']:
                    if hasattr(layer, attr):
                        var = getattr(layer, attr)
                        var.assign(tf.keras.initializers.Zeros()(var.shape, var.dtype))
            elif hasattr(layer, 'kernel_initializer') and layer.trainable:
                layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))
            if hasattr(layer, 'bias_initializer') and layer.trainable:
                layer.bias.assign(layer.bias_initializer(tf.shape(layer.bias)))

    # Create model fit and evaluate
    otrain, otest, ltrain, ltest, history = 0, 0, -1, -1, None # JLF 12/04/2024
    model = parameter
    Net = model_type(model, verbose=verbose)
    callback = TemperatureAnnealingCallback(10.0, 1e-6, model.epochs)
    callbacks = [callback] if temperature else None

    # fit
    v = True if verbose == 2 else False
    Niter = failure if failure else 10
    for niter in range(Niter): # seed training and eventually restart
        try:            
            history = Net.model.fit(Xtrain, Ytrain,
                                    validation_data=(Xtest, Ytest),
                                    epochs=int(Niter/10),
                                    batch_size=model.batch_size,
                                    callbacks=callbacks,
                                    verbose=v)
            # evaluate training set
            ytrain, stats = evaluate_model(Net.model, Xtrain, Ytrain,
                                       model, verbose=verbose)
            otrain, ltrain = stats.train_objective[0], stats.train_loss[0]
            if failure and otrain <= 0:
                reinitialize_weights(Net.model)
            else:
                break
        except:
            print(f'Failure number: {niter+1} R2: {otrain}')
            reinitialize_weights(Net.model)
        

    # complete training 
    history = Net.model.fit(Xtrain, Ytrain, 
                            validation_data=(Xtest, Ytest),
                            epochs=model.epochs,
                            batch_size=model.batch_size,
                            callbacks=callbacks,
                            verbose=v)

    # evaluate training set
    ytrain, stats = evaluate_model(Net.model, Xtrain, Ytrain,
                                   model, verbose=verbose)
    otrain, ltrain = stats.train_objective[0], stats.train_loss[0]
    
    # evaluate test set
    ytest, stats  = evaluate_model(Net.model, Xtest,  Ytest,
                                   model, verbose=verbose)
    otest, ltest = stats.test_objective[0], stats.test_loss[0]

    if verbose:
        if temperature:
            from Library.Build_Reservoir import identical_media
            X_true = parameter.res.X
            L = parameter.number_constraint+1
            X_pred = np.round(ytest[:,L:L+X_true.shape[1]], decimals=0)
            size = X_pred.shape[0]
            found = identical_media(X_true, X_pred)
        else:
            found, size = 0, 0
        print(f'train = {otrain:.2f} test = {otest:.2f} \
loss-train = {ltrain:.6f} loss-test = {ltest:.6f} \
Media found = {found} / {size}')

    return Net, ytrain, ytest, otrain, ltrain, otest, ltest, history

def train_evaluate_model(parameter, failure=0, temperature=0, verbose=False):
    # A standard function to create a model, fit, and Kflod cross validate
    # Kfold is performed for param.xfold test sets (if param.niter = 0)
    # otherwise only for niter test sets
    # Inputs:
    # - all necessary parameter including
    #   parameter.model, the function used to create the model
    #   parameter.input_model, the function used to shape the model inputs
    #   parameter.X and parameter.Y, the dataset
    #   parameter.scoring_function for regression or classification
    # Outputs:
    # - the best model (highest Q2/Acc on kfold test sets)
    # - the values predicted for each fold (if param.niter = 0)
    #   or the whole set when (param.niter > 0)
    # - the mean R2/Acc on the test sets
    # - the mean constraint value on the test sets
    # Must have verbose=True to verbose the fit 

    from sklearn.model_selection import KFold
    
    param = copy.copy(parameter)
    X, Y = model_input(param, verbose=verbose)
    param.X, param.Y = X, Y
    
    # Train on all data
    if param.xfold < 2:  # no cross-validation 
        if verbose: print(f'-------train {X.shape} {Y.shape}')
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param, X, Y, X, Y, 
                    failure=failure, temperature=temperature, verbose=verbose)
        # Return Stats
        stats = ReturnStats(otrain, 0, ltrain, 0, otest, 0, ltest, 0)
        return Net, ytrain, stats, history

    # Cross-validation loop
    Otrain, Otest, Ltrain, Ltest, Omax, Netmax, Ypred = \
    [], [], [], [], -1.0e32, None, np.copy(Y)
    kfold = KFold(n_splits=param.xfold, shuffle=True)
    kiter = 0
    for train, test in kfold.split(X, Y):
        if verbose: print(f'-------train {X[train].shape} {Y[train].shape}')
        if verbose: print(f'-------test  {X[test].shape} {Y[test].shape}')
        Net, ytrain, ytest, otrain, ltrain, otest, ltest, history = \
        train_model(param, X[train], Y[train], X[test], Y[test], 
                    failure=failure, temperature=temperature, verbose=verbose)
        # compile Objective (O) and Constraint (C) for train and test
        Otrain.append(otrain)
        Otest.append(otest)
        Ltrain.append(ltrain)
        Ltest.append(ltest)
        # in case y does not have the same shape than Y
        if Ypred.shape[1] != ytest.shape[1]:
            n, m = Y.shape[0], ytest.shape[1]
            Ypred = np.zeros(n * m).reshape(n, m)
        for i in range(len(test)): # Get Ypred for each fold
            Ypred[test[i]] = ytest[i]
        # Get the best network
        (Omax, Netmax) = (otest, Net) if otest > Omax else (Omax, Netmax)
        kiter += 1
        if (param.niter > 0 and kiter >= param.niter) or kiter >= param.xfold:
                break

    if param.niter > 0: # Prediction on the whole dataset
        Ypred, _ = evaluate_model(Netmax.model, X, Y, param, verbose=verbose)

    # Get Stats
    stats = ReturnStats(np.mean(Otrain), np.std(Otrain),
                        np.mean(Ltrain), np.std(Ltrain),
                        np.mean(Otest),  np.std(Otest),
                        np.mean(Ltest),  np.std(Ltest))

    return Netmax, Ypred, stats, history

################################################################################
# Create, save, load and print Neural Model
# ##############################################################################

from Library.Build_Dataset import TrainingSet
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
    
from sklearn.metrics import r2_score 
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
SCORING_FUNCTIONS = {
    'r2_score': r2_score,
    'accuracy_score': accuracy_score,
    'f1_score': f1_score,
    'matthews_corrcoef': matthews_corrcoef
}

class Neural_Model:
    # To save, load & print all kinds of models including reservoirs
    def __init__(self,
                 trainingfile=None, # training set parameter file
                 objective=False,
                 model=None, # the actual Keras model
                 model_type='AMN', # the function called Dense, AMN, RC
                 scaler=False, # X is not scaled by default
                 input_dim=0, output_dim=0, # model IO dimensions
                 n_hidden=0, hidden_dim=0, # default no hidden layer
                 activation='relu', # activation for last layer
                 # for all trainable models adam default learning rate = 1e-3
                 scoring_function=r2_score, 
                 epochs=0, train_rate=1e-3, dropout=0.25, batch_size=5,
                 niter=0, xfold=5, # Cross validation LOO does not work
                 verbose=False
                ):
        # Create empty object
        if model_type == '':
            return
        # model architecture parameters
        self.trainingfile = trainingfile
        self.model = model
        self.model_type = model_type
        self.scaler = float(scaler) # From bool to float
        self.input_dim = input_dim 
        self.output_dim = output_dim 
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.activation = activation
        # Training parameters
        self.epochs = epochs
        self.scoring_function = scoring_function
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        if 'AMN' in self.model_type:
            self.number_constraint = 3 # SV, Vin, Vko
        else:
            self.number_constraint = 0
        # Get additional parameters (matrices)
        self.get_parameter(init=True, objective=objective, verbose=verbose)
        
    def get_parameter(self, init=False, objective=False, verbose=False):
        from Library.Build_Dataset import TrainingSet, get_index_from_id
        # load parameter file if provided

        if self.trainingfile is None:
            return
        if not os.path.isfile(self.trainingfile+'.npz'):
            print(self.trainingfile+'.npz')
            sys.exit('parameter file not found')
        parameter = TrainingSet()
        parameter.load(self.trainingfile)
        if verbose == 2:
            print('1. in get_parameter parameter.Y.shape self.output_dim', 
                  parameter.Y.shape, self.output_dim)
            
        # Dealing with objective and Y filtering !!!!
        if init:
            if objective:
                self.objective = parameter.objective 
            else:
                self.objective = None
                
        if self.objective:
            parameter.filter_measure(measure=self.objective, verbose=verbose)
            self.Yall = parameter.Yall
        if verbose == 2:
            print('2. in get_parameter self.objective parameter.Y.shape self.output_dim', 
                  self.objective, parameter.Y.shape, self.output_dim)

        # matrices from parameter file 
        self.cobramodel = parameter.model
        self.medium = parameter.medium
        self.S = parameter.S # Stoichiometric matrix
        self.Pin = parameter.Pin # Boundary matrix from reaction to medium
        self.Pko = parameter.Pko # Matrix from reaction to ko        
        self.Pout = parameter.Pout # Measure matrix from reactions to measures
        self.X, self.Y = parameter.X, parameter.Y # Training set 
        
        # Update input_dim and output_dim
        self.input_dim = self.input_dim if self.input_dim > 0 else parameter.X.shape[1]
        self.output_dim = self.output_dim if self.output_dim > 0 else parameter.Y.shape[1]
        if verbose == 2:
            print('3. in get_parameter self.objective parameter.Y.shape self.output_dim', 
                  self.objective, parameter.Y.shape, self.output_dim)

        if self.input_dim != parameter.X.shape[1]:
            end = min(parameter.X.shape[1], self.input_dim)
            self.X = self.X[:,:end]
            self.input_dim = end
        if self.output_dim != parameter.Y.shape[1]:
            end = min(parameter.Y.shape[1], self.output_dim)
            self.Y = self.Y[:,:end]
            self.output_dim = end
        if verbose == 2:
            print('4. in get_parameter self.objective self.Y.shape self.output_dim', 
                  self.objective, self.Y.shape, self.output_dim)
            
    def save(self, filename, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        scoring_function_name = self.scoring_function.__name__
        s = str(self.trainingfile) + ","\
                    + str(self.model_type) + ","\
                    + str(self.number_constraint) + ","\
                    + str(self.objective) + ","\
                    + str(self.scaler) + ","\
                    + str(self.input_dim) + ","\
                    + str(self.output_dim) + ","\
                    + str(self.n_hidden) + ","\
                    + str(self.hidden_dim) + ","\
                    + str(self.activation) + ","\
                    + str(self.epochs) + ","\
                    + scoring_function_name + ","\
                    + str(self.train_rate) + ","\
                    + str(self.dropout) + ","\
                    + str(self.batch_size) + ","\
                    + str(self.niter) + ","\
                    + str(self.xfold)
        with open(fileparam, "w") as h:
            h.write(s)
        self.model.save(filemodel)

    def load(self, filename, output_dim=-1, verbose=False):
        fileparam = filename + "_param.csv"
        filemodel = filename + "_model.h5"
        if not os.path.isfile(fileparam):
            print(fileparam)
            sys.exit('parameter file not found')
        if not os.path.isfile(filemodel):
            print(filemodel)
            sys.exit('model file not found')
        # First read parameter file
        with open(fileparam, 'r') as h:
            for line in h:
                K = line.rstrip().split(',')
        # model architecture
        self.trainingfile = str(K[0])
        self.model_type = str(K[1])
        self.objective =  str(K[3])
        self.scaler = float(K[4])
        self.input_dim = int(K[5])
        self.output_dim = output_dim if output_dim > 0 else int(K[6])
        self.n_hidden = int(K[7])
        self.hidden_dim = int(K[8])
        self.activation = str(K[9])
        # Training parameters
        self.epochs = int(K[10])
        # Handle old 'regression' parameter for backward compatibility
        if K[11] in ['True', 'False']:
            self.scoring_function = r2_score if K[11] == 'True' else accuracy_score
        else:
            self.scoring_function = SCORING_FUNCTIONS[K[11]]
        self.train_rate = float(K[12])
        self.dropout = float(K[13])
        self.batch_size = int(K[14])
        self.niter = int(K[15])
        self.xfold = int(K[16])
        # Make objective a list
        self.objective = self.objective.replace('[', '')
        self.objective = self.objective.replace(']', '')
        self.objective = self.objective.replace('\'', '')
        self.objective = self.objective.replace("\"", "")
        self.objective = self.objective.split(',')
        if self.objective: # Dealing with empty objective !!!!
            if self.objective[0] == 'None':
                self.objective = None
        # Get additional parameters (matrices)
        if verbose == 2: 
            print('in loading before calling get_parameter self.objective, self.output_dim', 
                  self.objective , self.output_dim)
        self.get_parameter(verbose=verbose)
        if verbose == 2: 
            print('in loading after calling get_parameter self.objective, self.output_dim', 
                  self.objective , self.output_dim)
        self.Y = self.Y[:,:self.output_dim]
        # Then load model
        self.model = load_model(filemodel, compile=False)

    def printout(self, filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        print(f'training file: {self.trainingfile}')
        print(f'model type: {self.model_type}')
        print(f'model number constraints: {self.number_constraint}')
        print(f'model scaler: {self.scaler}')
        print(f'model input dim: {self.input_dim}')
        print(f'model output dim: {self.output_dim}')
        if self.trainingfile:
            if os.path.isfile(f'{self.trainingfile}.npz'):
                print(f'training set size {self.X.shape} {self.Y.shape}')
        else:
            print('no training set provided')
        if self.n_hidden > 0:
            print(f'nbr hidden layer: {self.n_hidden}')
            print(f'hidden layer size: {self.hidden_dim}')
            print(f'activation function: {self.activation}')
        if self.epochs > 0:
            print(f'training epochs: {self.epochs}')
            print(f'scoring function: {self.scoring_function.__name__}')
            print(f'training learn rate: {self.train_rate}')
            print(f'training dropout: {self.dropout}')
            print(f'training batch size: {self.batch_size}')
            print(f'training validation iter: {self.niter}')
            print(f'training xfold: {self.xfold}')
        if filename != '':
            sys.stdout.close()
            