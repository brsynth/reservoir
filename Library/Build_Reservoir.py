##############################################################################
# This library is making use of trained AMNs (cf. Build_Model.py) 
# in reseroir computing (RC). The reservoir (non-trainable AMN) 
# is squized between two standard ANNs. The purpose of the prior ANN is to 
# transform problem features into nutrients added to media. 
# The post-ANN reads reservoir output (user predefined specific 
# reaction rates) and produce a readout to best match training set values. 
# Note that the two ANNs are trained but not the reservoir (AMN). 
# Authors: Jean-loup Faulon, jfaulon@gmail.com 
# #############################################################################

from Library.Import import *

def my_tf_round(x, precision=1e-6):
    # return round(x) with precision
    # note tf.round is not differenciable
    xi = x * 1/precision
    xi = xi - tf.stop_gradient(xi - tf.round(xi))
    return xi * precision 
    
def input_RC(parameter, verbose=False):
    from sklearn.preprocessing import MinMaxScaler
    # Shape X and Y for RC
    scaler = MinMaxScaler()
    parameter.X = scaler.fit_transform(parameter.X)
    parameter.scaler = scaler
    if verbose: print(f'RC scaler: {parameter.scaler}')
    if verbose: print(f'RC input shape:{parameter.X.shape, parameter.Y.shape}')
    y = np.zeros(parameter.Y.shape[0]).reshape(parameter.Y.shape[0],1)
    # experimental dataset constraint
    parameter.Y = np.concatenate((parameter.Y, y), axis=1) 
    parameter.input_dim = parameter.X.shape[1]
    return parameter.X, parameter.Y

def distance_pred_true_media(tensor1_batch, tensor2_fixed):
    # A function to compute Tanimoto coefficients 
    # for a batch of N1 vectors against fixed N2 vectors

    def tanimoto_coefficient(A, B):
        # Tanimoto coefficient function
        AB = tf.reduce_sum(tf.multiply(A, B), axis=-1)
        AA = tf.reduce_sum(tf.multiply(A, A), axis=-1)
        BB = tf.reduce_sum(tf.multiply(B, B), axis=-1)
        return AB / (AA + BB - AB)

    # Reshape tensors for broadcasting
    tensor1_expanded = tf.expand_dims(tensor1_batch, 1)  # Shape (batch_size, 1, M)
    tensor2_expanded = tf.expand_dims(tensor2_fixed, 0)  # Shape (1, N2, M)
    # Compute the Tanimoto coefficient for each pair of vectors
    tanimoto_matrix = tanimoto_coefficient(tensor1_expanded, tensor2_expanded)
    # Compute 1 - Tanimoto coefficient
    one_minus_tanimoto_matrix = 1.0 - tanimoto_matrix
    # Find the minimum 1 - Tanimoto coefficient (smallest distance) 
    # for each vector in the batch
    min_one_minus_tanimoto = tf.reduce_min(one_minus_tanimoto_matrix, axis=1) 

    return min_one_minus_tanimoto

def gumbel_softmax(logits):
    # A differentiable one-hot
    # global_training_temperature is a global variable 
    gumbel_dist = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1)))
    y = logits + gumbel_dist
    y = tf.nn.softmax(y / global_training_temperature)
    return y

def get_reservoir_input(parameter, probs):
    # Transform probs into one-hot vector
    # Get the corresponding parameter.res.X vector
    from keras.layers import concatenate
    one_hot = gumbel_softmax(probs)
    X = parameter.res.X.astype(np.float32)
    result = tf.linalg.matmul(one_hot, X)
    result = concatenate([result, one_hot, probs], axis=1) # !!!   
    return result

def loss_reservoir_input(parameter, X_pred, outputs):
    # Return LossX the distance to parameter.res.X
    if parameter.weight_pred_true_media > 0:             
        X_true = tf.convert_to_tensor(np.float32(parameter.res.X))  
        LossX = distance_pred_true_media(X_pred, X_true)
        LossX = parameter.weight_pred_true_media * tf.reshape(LossX, [-1, 1])
    else:
        LossX = 0 * outputs  # zero vector
    return LossX

def RC_prior(Prior_inputs, parameter, verbose=False):
    # Prior trainable network that generates 
    # an output = input to the reservoir
    from Library.Build_Model import Dense_layers
    from keras.layers import Lambda
    
    if parameter.prior:
        activation = parameter.prior.activation
        if activation == 'gumbel_softmax': # selective prior
            parameter.prior.activation = 'relu' # relu better than softmax 
            Prior_outputs = Dense_layers(Prior_inputs, parameter.prior, 
                                         trainable=True, verbose=verbose)
            Prior_outputs = Lambda(lambda x: get_reservoir_input(parameter, x))(Prior_outputs)
            parameter.prior.activation = activation # reset
        else: # hard_sigmoid or relu...
            Prior_outputs = Dense_layers(Prior_inputs, parameter.prior, 
                                         trainable=True, verbose=verbose)
    else:
        Prior_outputs = Prior_inputs
        
    if verbose:
        print(f'Prior IO {Prior_inputs.shape}  {Prior_outputs.shape}')
        
    return Prior_outputs

def get_reservoir_matrix_bias(reservoir, verbose=False):
    # Get W and bias matrices extracted from the reservoir
    from keras.layers import Dense
    # Extract weights and biases from the pretrained res model
    layers = [layer for layer in reservoir.model.layers if isinstance(layer, Dense)]
    W = np.asarray([layer.get_weights()[0] for layer in layers], dtype=object)
    bias = np.asarray([layer.get_weights()[1] for layer in layers], dtype=object)
    if verbose:
        print(f'Reservoir matrices shapes W {W.shape} bias {bias.shape}')
    return W, bias
    
def RC_reservoir_matrix_bias(reservoir, inputs, mode, verbose=False):
    # Run non-trainable reservoir (must have been created and saved)
    
    if reservoir.scaler != 0: # Scaled to fit training data
        inputs = inputs / reservoir.scaler
        
    # Run reservoir just getting the weights
    outputs = inputs
    W, bias = get_reservoir_matrix_bias(reservoir, verbose=verbose)
    for i in range(len(W)):  # All layers
        w = tf.constant(W[i], dtype=tf.float32)
        b = tf.constant(bias[i], dtype=tf.float32)
        outputs = tf.nn.relu(tf.linalg.matmul(outputs, w, b_is_sparse=True) + b)
    if 'objective' in mode: # only objective otherwise all fluxes
        Pout  = tf.convert_to_tensor(np.float32(reservoir.Pout))
        outputs = tf.linalg.matmul(outputs, tf.transpose(Pout), b_is_sparse=True)
        
    return outputs
    
def RC_reservoir(Res_inputs, parameter, mask=None, verbose=False):
    # Run non-trainable reservoir (must have been created and saved)
    
    if parameter.mask_prior:  # O/1 mask
        inputs_mask = tf.math.divide_no_nan(mask, mask) 
        Res_inputs = tf.math.multiply(Res_input, inputs_mask)
    Res_outputs = RC_reservoir_matrix_bias(parameter.res, Res_inputs, 
                                           parameter.mode, verbose=verbose)    
    if verbose: 
        print(f'Res IO {Res_inputs.shape} {Res_outputs.shape}')
        
    return Res_outputs

def RC_post(Post_inputs, parameter, verbose=False):
    # Post trainable network that takes as input the reservoir output 
    # and produces the problem's output
    from Library.Build_Model import Dense_layers
    
    if parameter.precision > 0: # Rounding
        Post_inputs = my_tf_round(Post_inputs, precision=parameter.precision)
    if parameter.post: # Create a post ANN
        Post_outputs = Dense_layers(Post_inputs, parameter.post, verbose=verbose)
        if verbose: 
            print(f'Post IO {Post_inputs.shape} {Post_outputs.shape}')
    else: # Apply MinMax scaler on output
        Min, Max = np.min(parameter.res.Y), np.max(parameter.res.Y)
        Post_outputs = tf.subtract(Post_inputs, Min)
        Post_outputs = Post_outputs / (Max-Min)
        if verbose: 
            print(f'Post IO {Post_inputs.shape} {Post_outputs.shape} \
MinMax-Scaler: ({Min:.4f}, {Max:.4f})')
            
    return Post_outputs

def RC(parameter, verbose=False):
    # Build and return a Reservoir Computing model
    # The model is composed of
    # - A prior trainable network that generates 
    #   an output = input of the reservoir
    # - The non-trainable reservoir (must have been created and saved)
    # - A post trainable network that takes as input the reservoir output 
    #   and produces the problem's output
    # - A last layer that concatenates the prior trainable output 
    #   and the post trainable output
    
    from Library.Build_Model import my_mae, my_mse, CROP
    from Library.Build_Model import my_r2, my_acc, my_binary_crossentropy
    from keras.layers import Input, concatenate

    # Prior 
    inputs = Input(shape=(parameter.input_dim,))
    Prior_outputs =  RC_prior(inputs, parameter, verbose=verbose)

    # Reservoir 
    Res_inputs = CROP(1, 0, parameter.res.X.shape[1])(Prior_outputs)  
    Res_outputs = RC_reservoir(Res_inputs, parameter, inputs, verbose=verbose)
    
    # Post 
    Post_inputs = Res_outputs
    Post_outputs = RC_post(Post_inputs, parameter, verbose=verbose)

    # Compile 
    LossX =  loss_reservoir_input(parameter, Res_inputs, Post_outputs)
    outputs = concatenate([Post_outputs, LossX, Prior_outputs], axis=1)
    model = keras.models.Model(inputs, outputs)
    loss, metrics = (my_mse, [my_r2]) if parameter.regression \
    else (my_binary_crossentropy, [my_acc])
    opt = tf.keras.optimizers.Adam(learning_rate=parameter.train_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    if verbose == 2: print(model.summary())
    parameter.model = model

    return parameter

class RC_Model:
    # To save, load & print RC models
    def __init__(self, 
                 mode='AMN-objective', 
                 # modes are 
                 # AMN-objective: Regular reservoir trained on exprimental data
                 #                return only the objective value
                 # AMN-phenotype: Regular reservoir trained on exprimental data
                 #                return all fluxes
                 reservoirfile=None, # reservoir file (a Neural_Model)
                 precision=0,
                 X=[], # X training data
                 Y=[], # Y training data
                 model=None, # the actual Keras model
                 input_dim=0, output_dim=0, # model IO dimensions
                 # for prior network in RC model
                 # default is n_hidden_prior=-1: no prior network
                 n_hidden_prior=-1, hidden_dim_prior=-1, 
                 activation_prior='relu',
                 mask_prior=False,
                 # for post network in RC model
                 # defaulf is n_hidden_post=-1: no post network
                 n_hidden_post=-1, hidden_dim_post=-1, activation_post='linear', 
                 # for all trainable models adam default learning rate = 1e-3
                 regression=True, 
                 epochs=0, train_rate=1e-3, dropout=0.25, batch_size=5,
                 niter=0, xfold=5, # cross validation 
                 # weight to compute a loss between predicted media 
                 # and experimentam media in model.res
                 weight_pred_true_media=0, # use to maximize media already in training
                 verbose=False
                ):
        
        from Library.Build_Model import Neural_Model
        from sklearn.preprocessing import MinMaxScaler

        if len(X) < 1 or len(Y) < 1:
            sys.exit('must provide X and Y arrays') 
            
        # Training parameters apply MinMax scaler to X
        self.mode = mode
        self.reservoirfile = reservoirfile
        self.precision = precision
        scaler = MinMaxScaler()
        self.X = X 
        self.Y = Y
        self.number_constraint = 1
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.epochs = epochs
        self.regression = regression
        self.train_rate = train_rate
        self.dropout = dropout
        self.batch_size = batch_size
        self.niter = niter
        self.xfold = xfold
        self.mask_prior = mask_prior
        
        # Create reservoir
        self.res = Neural_Model()
        self.res.load(reservoirfile)
        self.res.get_parameter(verbose=verbose)
        
        # Get matrices for loss computation
        self.S = self.res.S # Stoichiometric matrix
        self.Pin = self.res.Pin # Boundary matrix from reaction to medium
        self.Pout = self.res.Pout # Measurement matrix from reactions to measures
    
        # Set RC model  
        self.model_type, self.prior, self.post = 'RC', None, None
        self.weight_pred_true_media = weight_pred_true_media

        # Set prior network
        if n_hidden_prior > -1: 
            output_dim = self.res.X.shape[0] \
            if activation_prior == 'gumbel_softmax' else self.res.input_dim
            self.prior = Neural_Model(model_type = 'ANN_Dense',
                                      input_dim = self.input_dim, 
                                      output_dim = output_dim,
                                      n_hidden = n_hidden_prior, 
                                      hidden_dim = hidden_dim_prior,
                                      activation = activation_prior)
        # Set post network input_dim = output_dim 
        # take as input the objective of the reservoir
        if n_hidden_post > -1:
            self.post = Neural_Model(model_type = 'ANN_Dense',
                                     input_dim = self.output_dim, 
                                     output_dim = self.output_dim,
                                     n_hidden = n_hidden_post, 
                                     hidden_dim = hidden_dim_post,
                                     activation = activation_post)
        
    def printout(self, filename=''):
        if filename != '':
            sys.stdout = open(filename, 'a')
        print(f'RC reservoir file: {self.reservoirfile}')
        print(f'RC model type: {self.model_type}')
        print(f'RC number constraint: {self.number_constraint}')
        print(f'RC precsion: {self.precision}')
        print(f'RC model input dim: {self.input_dim}')
        print(f'RC model output dim: {self.output_dim}')
        print(f'training set size {self.X.shape} {self.Y.shape}')
        if self.reservoirfile:
            print(f'reservoir S, Pin, Pout matrices {self.S.shape} {self.Pin.shape} {self.Pout.shape}')
        if self.epochs > 0:
            print(f'RC training epochs: {self.epochs}')
            print(f'RC training regression: {self.regression}')
            print(f'RC training learn rate: {self.train_rate}')
            print(f'RC training dropout: {self.dropout}')
            print(f'RC training batch size: {self.batch_size}')
            print(f'RC training validation iter: {self.niter}')
            print(f'RC training xfold: {self.xfold}')
        if self.prior:
            print('--------prior network --------')
            self.prior.printout(filename)
        print('--------reservoir network-----')
        self.res.printout(filename)
        if self.post:
            print('--------post network ---------')
            self.post.printout(filename)
        if filename != '':
            sys.stdout.close()

def identical_media(X_true, X_pred):
    # return Nbr of experimental media found
    X_true = np.round(X_true, decimals=0)
    X_pred = np.round(X_pred, decimals=0)
    m = 0
    for ip in range(X_pred.shape[0]):
        xp = list(X_pred[ip])
        for it in range(X_true.shape[0]):
            xt = list(X_true[it])
            if xp == xt:
                m += 1
    return m
    
def RC_run(reservoirfile, X, Y, 
           mode='AMN_objective',
           regression=True, 
           n_hidden_prior=1, hidden_dim_prior=28, 
           n_hidden_post=-1, hidden_dim_post=0, 
           activation_prior='sharp_sigmoid',
           precision=0, train_rate=1.0e-3, xfold=5, epochs=100, repeat=1,
           temperature=0, failure=0, weight_pred_true_media=0, 
           seed=1, verbose=False):
    # Train and cross validate with a reservoir
    # cf. class RC_Model for parameter definitions
    
    from Library.Build_Model import train_evaluate_model, evaluate_model, model_input
    from sklearn.metrics import r2_score, accuracy_score
    
    Maxloop, R2, Q2, PRED, Q2best, mbest = repeat, [], [], [], -1e6, 0
    batch_size = 100 if X.shape[0] > 1000 else 10

    for Nloop in range(Maxloop):
        # Create model
        model = RC_Model(mode=mode,
                         reservoirfile=reservoirfile,
                         X=X, Y=Y,
                         regression=regression,
                         precision=precision,
                         n_hidden_prior=n_hidden_prior,                  
                         n_hidden_post=n_hidden_post,
                         hidden_dim_prior=hidden_dim_prior, 
                         hidden_dim_post=hidden_dim_post, 
                         activation_prior=activation_prior,
                         activation_post='relu',
                         batch_size=batch_size,
                         epochs=epochs, 
                         train_rate=train_rate, 
                         xfold=xfold, 
                         weight_pred_true_media=weight_pred_true_media,                      
                         verbose=verbose)
        if verbose: model.printout()
            
        # Train and evaluate
        reservoir, pred, stats, _ = train_evaluate_model(model, 
                                                         temperature=temperature,
                                                         failure=failure,
                                                         verbose=verbose)
        R2.append(stats.train_objective[0])
        r2 = r2_score(Y, pred[:, 0], multioutput='variance_weighted')
        Xe = model.res.X
        L = model.number_constraint+1
        Xr = np.round(pred[:,L:L+Xe.shape[1]], decimals=0)
        m = identical_media(Xr, Xe)
        if verbose:
            print(f'iteration: {Nloop} q2: {r2:.4f} identical predicted vs. measured media: {m} / {Xr.shape[0]}')
        if r2 > Q2best:
            Q2best, mbest, Nbest, modelbest, statsbest = r2, m/Xr.shape[0], Nloop, model, stats
        Q2.append(r2)
        PRED.append(pred)  
        
    # Compute stats
    Q2, PRED = np.asarray(Q2), np.asarray(PRED)
    model, pred = modelbest, PRED[Nbest]
    R2_avr, R2_dev =  np.mean(R2), np.std(R2)
    Q2_avr, Q2_dev =  np.mean(Q2), np.std(Q2)
    if verbose:
        r2 = r2_score(Y, pred[:, 0], multioutput='variance_weighted')
        Loss_X = pred[:, 1]
        Xe = model.res.X
        L = model.number_constraint+1
        Xr = np.round(pred[:,L:L+Xe.shape[1]], decimals=0)
        m = identical_media(Xr, Xe)
        print(f'Average R2 {R2_avr:.4f}±{R2_dev:.4f}')
        print(f'Average Q2 {Q2_avr:.4f}±{Q2_dev:.4f}')
        print(f'Best model Q2 {r2:.4f} identical predicted vs. measured media: {mbest*Xr.shape[0]} / {Xr.shape[0]}')
        print(f'Loss X average {np.mean(Loss_X):.4f} max {np.max(Loss_X):.4f}')

    return model, pred, R2_avr, R2_dev, Q2_avr, Q2_dev, mbest

def RC_write_multiple(reservoirfile, resultfile, 
                      model, y_true, pred,  
                      multiple=0, 
                      precision_X=0, # X are integer when > 0
                      precision_Y=0, # Rounding when > 0
                      verbose=True):  
    # Save results with multiple reservoir
    
    from Library.Build_Model import Neural_Model
    from Library.Build_Model import train_evaluate_model, evaluate_model, model_input
    from Library.Build_Dataset import get_minmed_varmed_ko
    from sklearn.metrics import r2_score, accuracy_score

    if multiple < 1:
        return
    medium = model.res.medium
    minmed, varmed, ko = get_minmed_varmed_ko(medium)
    medium = list(varmed.keys())
    medium.append('Y_true')
    medium.append('Y_pred')
    medium.append('Growth_ref')
    medium.append('Growth_avr')
    medium.append('Growth_dev')
    medium.append('Growth_min')
    medium.append('Growth_max')
    y_pred = pred[:,0].reshape(-1, 1)
    L = model.number_constraint+1
    Xe = model.res.X
    Xr = pred[:,L:L+Xe.shape[1]]
    Xr = np.round(Xr, decimals=0) if precision_X else Xr
     
    # Get growth rate with X = Xin and multipe reservoirs
    # i=0 is the reservoir used to compute y_pred
    reservoir = Neural_Model()
    Y_growth = []
    R2best, Nbest = -10, 0
    for i in range(int(multiple)):
        resfi = f'{reservoirfile}_{str(i)}' if multiple > 1 else reservoirfile
        reservoir.load(resfi)
        reservoir.get_parameter(verbose=verbose)
        Res_inputs = tf.constant(Xr, dtype=tf.float32)
        pred = RC_reservoir_matrix_bias(reservoir, Res_inputs, model.mode).numpy()
        y_growth = pred[:,0].reshape(-1, 1) 
        if precision_Y > 0:
            y_growth = np.round(y_growth/precision) * precision
        r2 = r2_score(y_true, y_growth , multioutput='variance_weighted')
        if r2 > R2best:
            Nbest, R2best = i, r2
        if verbose: 
            print(f'Load reservoir {resfi} type: {reservoir.model_type} r2 (y_growth vs. y_true): {r2:.4f}')
        Y_growth.append(y_growth)
    if verbose: 
        print(f'Best model number {Nbest} r2 (y_growth vs. y_true): {R2best:.4f}')
    Y_growth = np.asarray(Y_growth)
    y_growth_ref = Y_growth[0].reshape(-1, 1) # reference = 0 the one used to compute y_pred
    y_growth_avr = np.mean(Y_growth, axis=0).reshape(-1, 1)
    y_growth_dev = np.std(Y_growth, axis=0).reshape(-1, 1)
    y_growth_min = np.min(Y_growth, axis=0).reshape(-1, 1)
    y_growth_max = np.max(Y_growth, axis=0).reshape(-1, 1)
    H = np.asarray(medium)
    D = np.concatenate((Xr, y_true, y_pred, y_growth_ref, y_growth_avr, 
                        y_growth_dev, y_growth_min, y_growth_max), axis=1)
    write_csv(resultfile, H, D)
