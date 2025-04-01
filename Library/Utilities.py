###############################################################################
# This file iprovide general utilities to read files using panda
# and perform ML with sklearn
# Author: Jean-loup Faulon jfaulon@gmail.com
# Updates: 24/11/2023
###############################################################################

from Library.Import import *

###############################################################################
# Reading and writting files with panda
###############################################################################

def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    filename += '.csv'
    dataframe = pd.read_csv(filename, header=0)
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:,:])
    return HEADER, DATA

def write_csv(filename, HEADER, DATA):
    # write a csv file
    filename += '.csv'
    with open(filename, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(HEADER)
        # write the data
        writer.writerows(DATA)

from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range = (0,1))
scalerY = MinMaxScaler(feature_range = (0,1))
    
def read_XY(filename, nY=1, scaling=''):
    # Format data for training
    # Function read_training_data is defined in module (1)
    # if scaling == 'X' X is scaled
    # if scaling == 'Y' Y is scaled
    # if scaling == 'XY' X and Y are scaled
    H, XY = read_csv(filename)
    XY = np.asarray(XY)
    XY = XY[~np.isnan(XY).any(axis=1)]
    nX = XY.shape[1]-nY
    X = XY[ : ,    : nX]
    Y = XY[ : , nX : nX+nY] if nY else []
    if scaling == 'X' or scaling == 'XY':
        X = scalerX.fit_transform(X)
    if scaling == 'Y' or scaling == 'XY':
        Y = scalerY.fit_transform(Y)
    return H, X, Y

def write_XY(filename, H, X, Y):
    # Write file first scaling back
    # Function read_training_data is defined in module (1)
    X = scalerX.inverse_transform(X)
    Y = scalerY.inverse_transform(X)
    D = np.concatenate(X, Y, axis=1)
    write_csv(filename, H, D)

def MaxScaler(data, Max_Scaler = 1.0e12):
    # Max standardize np array data
    if Max_Scaler == 1.0e12: # Scale
        Max_Scaler = np.max(data)
        data = data/Max_Scaler
    else: # Descale
        data = data * Max_Scaler
        Max_Scaler = 1.0e12      
    return data, Max_Scaler

###############################################################################
# Simple sklearn models 
###############################################################################

def bayes_classifier(X, y, regression=False):
    # multiclass classification with naive_bayes
    if regression:
        sys.exit('No regression with bayes_classifier') 
    from sklearn.naive_bayes import GaussianNB 
    gnb = GaussianNB().fit(X, y)
    return gnb

def svm_classifier(X, y, regression=False):
    # multiclass classification with svm
    if regression:
        sys.exit('No regression with svm_classifier') 
    from sklearn.svm import SVC 
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X, y) 
    return svm_model_linear
    
def decision_tree_classifier(X, y, regression=False):
    # multiclass classification with DescisionTreeClassifier 
    if regression:
        sys.exit('No regression with decision_tree_classifier') 
    from sklearn.tree import DecisionTreeClassifier 
    dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X, y) 
    return dtree_model

def GP(X, y, regression=False):
    # Regression or classification with GaussianProcess
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
    if regression:
        kernel = DotProduct() + WhiteKernel()
        gp_model = GaussianProcessRegressor(kernel=kernel,
                                random_state=0).fit(X, y)
    else:
        kernel = 1.0 * RBF(1.0)
        gp_model = GaussianProcessClassifier(kernel=kernel,
                                             # optimizer=None, 
                                             random_state=0).fit(X, y)
    return gp_model

def XGB(X, y, regression=False):
    # Regression or classification with a XGBoost
    from xgboost import XGBRegressor
    from xgboost import XGBClassifier
    if regression:
        xgb_model = XGBRegressor().fit(X, y)
    else:
        xgb_model = XGBClassifier().fit(X, y)
    return xgb_model

def MLP(X, y, regression=False):
    # Regression or classification with Multilinear perceptron
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    hidden_size = 16 if X.shape[1] == 2 else 2*int(X.shape[1]) 
    if regression:
        model = MLPRegressor(hidden_layer_sizes = (hidden_size),
                             solver ='adam', 
                             max_iter=10000,
                             early_stopping = True,
                             learning_rate='adaptive')
    else:
        model = MLPClassifier(hidden_layer_sizes = (hidden_size),
                             solver ='adam', 
                             max_iter=10000,
                             early_stopping=False,
                             learning_rate='adaptive')
    model.fit(X, y.flatten())
    return model

def Linear(X, y, regression=False):
    # Regression with LinearRegression
    from sklearn.linear_model import LinearRegression
    if regression == False:
        sys.exit('No classification with LinearRegression') 
    model = LinearRegression().fit(X, y)
    return model

###############################################################################
# LeaveXout functions 
###############################################################################

def best_accuracy_threshold(y_pred, y_true):
    # Get best accuracy moving a treshold 
    # best splitting labels
    from sklearn.metrics import roc_curve, accuracy_score

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Compute Youden's J statistic for each threshold
    youden_j = tpr - fpr

    # Find the threshold that maximizes Youden's J statistic
    best_threshold_index = np.argmax(youden_j)
    best_threshold = thresholds[best_threshold_index]

    # Optionally, determine the direction for comparison
    direction = "less than" \
    if np.mean(y_pred[y_true == 0]) < np.mean(y_pred[y_true == 1]) \
    else "greater than"

    # Compute accuracy for each threshold
    accuracies = []
    for threshold in thresholds:
        if direction == "less than":
            y_pred_labels = (y_pred < threshold).astype(int)
        else:
            y_pred_labels = (y_pred > threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred_labels)
        accuracies.append(accuracy)

    # Find the threshold that gives the highest accuracy
    best_accuracy_index = np.argmax(accuracies)
    best_accuracy_threshold = thresholds[best_accuracy_index]
    best_accuracy = accuracies[best_accuracy_index]

    return best_accuracy, best_accuracy_threshold


from sklearn.metrics import r2_score
def LXO(X, y, 
        learner=Linear, 
        scoring_function=r2_score, 
        xfold=10, 
        seed=0,
        verbose=False):
    # Local function: perform LXO with provided learner
    
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import LabelEncoder
    
    regression = True if scoring_function==r2_score else False
    le = LabelEncoder()
    y_pred = 0 * y
    if xfold < 2:
        # y = np.asarray(y).reshape(-1,1)
        y = y if regression else le.fit_transform(y)
        model = learner(X, y, regression=regression)
        y_pred = model.predict(X)
        y_pred = y_pred if regression else le.inverse_transform(y_pred.ravel())
    else:
        kfold = KFold(n_splits=xfold, shuffle=True, random_state=seed)
        for train, test in kfold.split(X, y):
            y_train = y[train] if regression else le.fit_transform(y[train])
            model = learner(X[train], y_train, regression=regression)
            y_pred_test = model.predict(X[test])
            y_pred_test = y_pred_test.ravel() if regression \
            else le.inverse_transform(y_pred_test.ravel())
            y_pred_test = y_pred_test.reshape(-1,1)
            for i in range(len(test)):
                y_pred[test[i]] = y_pred_test[i]
                
    if verbose == 2:
        print(f'score: {scoring_function(y, y_pred):.2f}')
        
    return y_pred
    
def LeaveXout(X, y, F, learner=Linear, scoring_function=r2_score,
              xfold=10, niter=5, selection=0, verbose=False):
    # Perform LXO with feature selection removing features one at a time
    # Selection is the minimum number of features selected
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor

    regression = True if scoring_function==r2_score else False
    F = np.asarray(F)
    
    if selection > 0:
        
        # First reduce: Use RandomForest to rank features by importance
        if regression:
            rf = RandomForestRegressor(random_state=42)
        else:
            rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
    
        # Select the top features based on importance 
        # Select top 10*selection features
        top_features_idx = np.argsort(importances)[-10*selection:]  
        X = X[:, top_features_idx]
        F = F[top_features_idx]
        
        # Remove feature one at a time
        # Keep best score
        SCORE_BEST, F_BEST, scores = float('-inf'), F, []
        while X.shape[1] > selection:
            score_best = float('-inf')
            for I in range(X.shape[1]):
                XX = np.delete(X, I, 1)
                FF = np.delete(F, I, 0)
                scores = []
                for i in range(niter):
                    y_pred = LXO(XX, y.ravel(), learner=learner,
                        xfold=xfold, scoring_function=scoring_function, 
                        seed=i, verbose=verbose)
                    score = scoring_function(y, y_pred)
                    scores.append(score)
                score = np.mean(np.asarray(scores))
                stdev = np.std(np.asarray(scores))
                if score > score_best:
                    score_best, score_dev = score, stdev
                    I_best, X_best, F_best = I, np.copy(XX), np.copy(FF)     
            if verbose:
                print(f'Size: {X_best.shape[1]} Remove: {F[I_best]} '
                      f'Score: {score_best:.3f}±{score_dev:.3f}')
            X, F = X_best, F_best
            if score_best > SCORE_BEST:
                SCORE_BEST, X_BEST, F_BEST = \
                score_best, X_best, F_best  
        X, F = X_BEST, F_BEST
        
    # Compute final score
    scores, y_pred = [], {}
    for i in range(10*niter):
        y_pred[i] = LXO(X, y.ravel(), learner=learner,
                        xfold=xfold, scoring_function=scoring_function, 
                        seed=i, verbose=verbose)
        score = scoring_function(y, y_pred[i])
        scores.append(score)
        
    score_avr = np.mean(np.asarray(scores))
    score_dev = np.std(np.asarray(scores))
    y_pred = np.asarray(list(y_pred.values()))
    y_pred_avr = np.mean(y_pred, axis=0)
    y_pred_dev = np.std(y_pred, axis=0)
    if verbose > 1:
        print('y_true y_pred_avr±y_pred_dev')
        for i in range(y.shape[0]):
            print(f'{y[i]} {y_pred_avr[i]}±{y_pred_dev[i]}')        
    return score_avr, score_dev, F


###############################################################################
# Binarize feature function
###############################################################################

def map_and_binarize_feature(Xf, yf, Xr, yr, threshold, 
                             regression=True, verbose=False):
    # Binarize Xf features and rank features Xf and Xr using RandomForest
    # Map feature Xf to Xr according to ranking
    # If the number of features Nr is greater than 
    # the number of features Nf,  additional features 
    # are generated duplicating features in Xf.
    # Finally feature Xf are binarized.
    # Parameters:
    # threshold: The number threshold for binarization
    # Returns: A binary version of the transformed feature Xf.
    from sklearn.preprocessing import Binarizer
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier   
    from sklearn.inspection import permutation_importance   
    
    def rank_feature(X, y, regression=True):
        # Rank features based on permutation_importance
        # using random forest regressor
        if regression:
            model = RandomForestRegressor(n_estimators=100, 
                                          random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, 
                                           random_state=42)
            
        model.fit(X, y)
        
        # Compute permutation importance
        perm_importance = permutation_importance(model, X, y, 
                                                 n_repeats=10, random_state=42)
        
        # Create a dataframe to hold feature names and 
        # their permutation importance scores
        feature_names = list(range(X.shape[1]))
        perm_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': perm_importance.importances_mean
            })
        
        # Sort by importance
        perm_importance_df = perm_importance_df.sort_values(by='Importance', 
                                                            ascending=False)
        sorted_features = perm_importance_df['Feature'].tolist()
        
        return sorted_features

    Nf, Nr = Xf.shape[1], Xr.shape[1]
    Xt, yt = np.copy(Xf), yf
    
    # Add columns to Xt to increase the number of features to Nr
    while Xt.shape[1] < Nr:
        i = np.random.randint(0, Xt.shape[1])
        Xt = np.concatenate((Xt, Xt[:,i].reshape(-1,1)), axis=1)
        
    # Binarize Xt data
    scaler = MinMaxScaler()
    Xt = scaler.fit_transform(Xt)  
    binarizer = Binarizer(threshold=threshold)
    Xb, yb = binarizer.fit_transform(Xt), yt

    # Rank the features of both Xb and Xr
    lb = rank_feature(Xb, yb.ravel(), regression=regression)
    lr = rank_feature(Xr, yr.ravel(), regression=True)
    if verbose:
        print('reservoir:', lr)
        print('problem  :', lb)
        
    # One-to-one mapping of ranked features  
    Xt = np.zeros((Xb.shape[0], Xr.shape[1]))
    for i in range(len(lr)):
        Xt[:,lr[i]] = Xb[:,lb[i]]
            
    return Xt