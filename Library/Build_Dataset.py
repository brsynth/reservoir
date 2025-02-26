###############################################################################
# This library create training sets for AMN-QP
# Training sets are either based on experimental datasets
# or FBA (Cobrapy) simulations
# The file also provides functions to run Cobra
# Authors: Jean-loup Faulon jfaulon@gmail.com
# Updates: - Nov 2023 to compute fix and variable medium with EXP datasets
#          - June 2024 to add kos with EXP datasets           
###############################################################################

from Library.Import import *
import cobra
import cobra.manipulation as manip
from cobra import Reaction, Metabolite, Model
from cobra.flux_analysis import pfba
from sklearn.utils import shuffle

###############################################################################
# Cobra's model utilities and matrices 
###############################################################################

def get_gene_id_from_name(model, gnames):
    # Return the gene IDs of the gene names
    # provided in gnames
    gids = []
    for gname in gnames:
        for g in model.genes:
            if g.name == gname:
                gids.append(g.id)
    return gids

def get_index_from_id(idname,L):
    # Return index in L of id name
    for i in range(len(L)):
        if L[i].id == idname:
            return i
    return -1

def get_objective(model):
    # Get the reaction carring the objective
    # Someone please tell me if there is
    # a clearner way in Cobra to get
    # the objective reaction

    r = str(model.objective.expression)
    r = r.split()
    r = r[0].split('*')
    obj_id = r[1]

    # line below crash if does not exist
    r = model.reactions.get_by_id(obj_id)

    return obj_id

def get_matrices(model, medium, ko, measure, reactions):
    # Get matrices for AMN_QP 
    # Return
    # - S [mxn]: stochiometric matrix
    # - Pin [n_in x n]: to go from reactions to medium fluxes
    # - Pko [n_ko x n]: to go from reactions to ko gene fluxes
    # - Pout [n_out x n]: to go from reactions to measured fluxes

    # m = metabolite, n = reaction/v/flux, p = medium
    S = np.asarray(cobra.util.array.create_stoichiometric_matrix(model))
    n, m = S.shape[1], S.shape[0]
    n_in, n_ko, n_out = len(medium), len(ko), len(measure)

    # Boundary matrices from reaction to medium fluxes
    Pin, i = np.zeros((n_in,n)), 0
    for rid in medium:
        j = get_index_from_id(rid,reactions)
        Pin[i][j] = 1
        i = i+1
        
    # Boundary matrices from reaction to ko fluxes
    Pko, i = np.zeros((n_ko,n)), 0
    for rid in ko:
        j = get_index_from_id(rid,reactions)
        Pko[i][j] = 1
        i = i+1
        
    # Experimental measurements matrix from reaction to measured fluxes
    Pout, i = np.zeros((n_out,n)), 0
    for rid in measure:
        j = get_index_from_id(rid,reactions)
        Pout[i][j] = 1
        i = i+1

    return S, Pin, Pko, Pout
            
###############################################################################
# Running Cobra
###############################################################################

def get_minmed_varmed_ko(medium, verbose=False):
    minmed, varmed, ko = {}, {}, {}
    for m in medium.keys():
        (l, v) = medium[m]
        if l == 0:
            ko[m] = (l, v)
        elif l == 1:
            minmed[m] = (l, v)
        elif l > 1:
            varmed[m] = (l, v)
    return minmed, varmed, ko
    
def print_medium(medium, cobra_min_flux=1.0e-8, verbose=False):
    if verbose > 1:
        for m, v in medium.items():
            if v > cobra_min_flux:
                print(f'cobra medium: {m} v: {v:.2f}')

def run_cobra(model,
              objective,
              medium,
              method='FBA',
              genekos=[],
              objective_fraction=0.75,
              cobra_min_flux=1.0e-8,
              verbose=False):
    # Inputs:
    # - model
    # - objective: a list of reactions (first two only are considered)
    # - medium: upper flux values for minimal and variable media
    # - method: FBA or pFBA
    # - genekos: a list of gene to KO
    # run FBA optimization to compute reaction fluxes on the provided model
    # set the medium using values in dictionary IN.
    # When 2 objectives are given one first maximize the first objective (obj1).
    # then one set the upper and lower bounds for that objective to
    # objective_fraction * obj1 (e.g. objective_fraction = 0.75) and maximize
    # for the second objective
    # Outputs:
    # - V: the reaction fluxes computed by FBA for all reactions
    # - V_obj: the flux value for the objective

    import signal
    def handle_timeout(sig, frame):
        raise TimeoutError('took too long')
        
    import warnings
    warnings.filterwarnings("ignore")

    # set the medium and objective
    modelmedium = model.medium.copy() 
    med = model.medium  # med will be modified
    for k in med.keys():  # Reset the medium
        med[k] = 0
    for k in medium.keys():  # Additional cmpds added to medium
        if k in med.keys():
            med[k] = float(medium[k])
    model.medium = med
    print_medium(model.medium, cobra_min_flux=cobra_min_flux, verbose=verbose)
        
    # set KOs
    kos = get_gene_id_from_name(model, genekos)
    for ko in kos:
        if verbose:
            print(f'KO gene name {genekos} id {ko}')
        model.genes.get_by_id(ko).knock_out()

    # run FBA for primal objective
    V = {x.id: 0 for x in model.reactions}
    model.objective = objective[0]
    if verbose > 1:
        print(f'Start Cobra Objectif = {objective}')    
    timeout = 1
    signal.signal(signal.SIGALRM, handle_timeout)
    try: 
        signal.alarm(timeout)
        solution = cobra.flux_analysis.pfba(model) \
        if method == 'pFBA' else model.optimize()
        V_obj = solution.fluxes[objective[0]]
        V_obj = 0 if V_obj < cobra_min_flux else V_obj    
        signal.alarm(0)
    except Exception as e: 
        V_obj = 0
        signal.alarm(0)
    if verbose > 1:
        print(f'End Cobra Objectif = {objective}, {method}, {V_obj}')

    # get the fluxes for all model reactions
    if V_obj:
        for x in model.reactions:
            V[x.id] = solution.fluxes[x.id]
            if math.fabs(float(V[x.id])) < cobra_min_flux:  # !!!
                V[x.id] = 0
    model.medium = modelmedium # reset to initial medium
 
    return V, V_obj

import threading
import warnings
import math

def run_cobra(model,
              objective,
              medium,
              method='FBA',
              genekos=[],
              objective_fraction=0.75,
              cobra_min_flux=1.0e-8,
              verbose=False):
    """
    Run FBA optimization to compute reaction fluxes on the provided model.
    
    Inputs:
    - model: COBRA model object
    - objective: a list of reactions (first two only are considered)
    - medium: upper flux values for minimal and variable media
    - method: 'FBA' or 'pFBA'
    - genekos: a list of genes to knock out
    - objective_fraction: fraction of the first objective to maintain
    - cobra_min_flux: minimum flux threshold
    - verbose: verbosity level
    
    Outputs:
    - V: reaction fluxes computed by FBA
    - V_obj: flux value for the objective
    """

    warnings.filterwarnings("ignore")

    # Set the medium and objective
    modelmedium = model.medium.copy() 
    med = model.medium  # Modify medium
    for k in med.keys():  # Reset the medium
        med[k] = 0
    for k in medium.keys():  # Add specified compounds to medium
        if k in med.keys():
            med[k] = float(medium[k])
    model.medium = med

    print_medium(model.medium, cobra_min_flux=cobra_min_flux, verbose=verbose)
    
    # Apply gene knockouts
    kos = get_gene_id_from_name(model, genekos)
    for ko in kos:
        if verbose:
            print(f'KO gene name {genekos} id {ko}')
        model.genes.get_by_id(ko).knock_out()

    # Run FBA for the primary objective
    V = {x.id: 0 for x in model.reactions}
    model.objective = objective[0]

    if verbose > 1:
        print(f'Start Cobra Objective = {objective}')    

    # Define a function to stop execution if it takes too long
    def timeout_handler():
        raise TimeoutError("Optimization took too long")

    # Set up a timer (1 second timeout)
    timeout = 1
    timer = threading.Timer(timeout, timeout_handler)

    try: 
        timer.start()  # Start timer
        solution = cobra.flux_analysis.pfba(model) if method == 'pFBA' else model.optimize()
        V_obj = solution.fluxes[objective[0]]
        V_obj = 0 if V_obj < cobra_min_flux else V_obj    
    except Exception as e: 
        V_obj = 0
        if verbose:
            print(f"Error during optimization: {e}")
    finally:
        timer.cancel()  # Stop the timer

    if verbose > 1:
        print(f'End Cobra Objective = {objective}, {method}, {V_obj}')

    # Get the fluxes for all model reactions
    if V_obj:
        for x in model.reactions:
            V[x.id] = solution.fluxes[x.id]
            if math.fabs(float(V[x.id])) < cobra_min_flux:
                V[x.id] = 0

    # Reset medium to initial state
    model.medium = modelmedium  
 
    return V, V_obj

###############################################################################
# Generating random medium runing Cobra
###############################################################################

def create_random_medium_cobra(model, objective, 
                               medium, mediumexp, ratmed,
                               method='FBA', 
                               genekos=[],
                               cobra_min_objective=1.0e-3,
                               cobra_max_objective=3.0,
                               verbose=False):
    
    # Generate a random input and get Cobra's output
    # Input:
    # - model
    # - objective: the reaction fluxes to optimize
    # - medium: list of reaction fluxes in medium
    # - mediumexp: experimental medium list
    # - method: the method used by Cobra
    # - genekos: a list of gene to KO
    # Make sure the medium does not kill the objective
    # i.e. objective > cobra_min_objective
    # Ouput:
    # - valmed: medium values

    minmed, varmed, ko = get_minmed_varmed_ko(medium)
   
    # ON = actual number of medium turned ON
    if len(mediumexp) > 0:
        ON = len(mediumexp)
    else:
        ON = np.random.randint(1, high=ratmed+1) # just a random integer ≤ ratmed
        # ON = np.random.binomial(len(varmed), ratmed, 1)[0] if ratmed < 1 else int(ratmed)
        # ON = 1 if ON == 0 else ON
    if verbose:
        print(f'ON: {ON}')
    
    # Get medium indices in varmed
    varid = list(set(varmed.keys()) & set(mediumexp)) if len(mediumexp) \
    else list(set(varmed.keys())) 
    valmed = {}
    
    # minimal medium
    for m in minmed.keys():
        (l, v) = minmed[m]
        valmed[m] = v 
        
    # create random medium choosing X fluxes in varmed at random
    varid = shuffle(varid) # that's where random choice occur
    varon = varid[:ON]
    for m in varon: # variable medium
        (l, v) = varmed[m]
        valmed[m] = np.random.randint(0,high=l+1) * v/l # can be zero
        if verbose:
            print(f'valmed: {m} level: {l}, value: {v}')

    # Run cobra
    _, obj = run_cobra(model, objective, valmed, genekos=genekos,
                       method=method, verbose=verbose)
    if obj < cobra_min_objective and verbose > 1:
        print(f'Cobra objective is zero')
    if obj > cobra_max_objective and verbose > 1:
        print(f'Cobra objective is over {cobra_max_objective}') 
         
    return valmed

def get_io_cobra(model, objective, 
                 medium, mediumexp, ratmed,
                 valmed={},
                 method='FBA',
                 genekos=[],
                 cobra_min_objective=1.0e-3,
                 cobra_max_objective=3.0,
                 verbose=False):
    # Generate a random input and get Cobra's output
    # Input:
    # - model: the cobra model
    # - objective: the list of objectiev fluxes to maximize
    # - medium: list of reaction fluxes in medium
    # - mediumexp: experimental medium list
    # - valmed: medium values
    # - method: the method used by Cobra
    # - genekos: a list of gene to KO
    # Output:
    # - X = medium fluxes , Y = fluxes 

    minmed, varmed, ko = get_minmed_varmed_ko(medium)
    X = len(varmed.keys()) * [0]
    Y = len(model.reactions) * [0]    
    
    if len(valmed) == 0:
        valmed = create_random_medium_cobra(model, objective,
                                            medium, mediumexp, ratmed,
                                            genekos=genekos, method=method,
                                            cobra_min_objective=cobra_min_objective,
                                            cobra_max_objective=cobra_max_objective,
                                            verbose=verbose)
        
    # Get valmed values in X
    for i in range(len(list(varmed.keys()))):
        m = list(varmed.keys())[i]
        if m in valmed.keys():
            X[i] = valmed[m]
            
    # Run Cobra
    V, obj = run_cobra(model, objective, valmed, genekos=genekos, 
                       method=method, verbose=verbose)
    if obj > cobra_min_objective and obj < cobra_max_objective:
        Y = list(V.values())
        if verbose:
            print(f'obj {obj} PASS MinMax ' 
                  f'[{cobra_min_objective, cobra_max_objective}] '
                  f'with provided medium: {valmed}')
    else:
        if verbose:
            print(f'obj {obj} FAIL MinMax ' 
                  f'[{cobra_min_objective, cobra_max_objective}] '
                  f'with provided medium: {valmed}')

    return np.asarray(X) , np.asarray(Y) 

def create_medium_run_cobra(model, objective, medium, X,  
                            method='FBA', scaler=1, 
                            genekos=[], verbose=False):
    # Create medium from X and run cobra
    # Input:
    # - model: the cobra model
    # - objective: the list of objective fluxes to maximize
    # - medium: list of reaction fluxes in medium
    # - X, y: training set where X is used to feed medium values
    # - scaler : to scale X values
    # - method: the method used by Cobra
    # - genekos: a list of gene to KO
    # Output:
    # - Y, V = objective, fluxes 
    
    minmed, varmed, ko = get_minmed_varmed_ko(medium)
    L, obj, V = X.shape[0], {}, {}
    
    for i in range(L):
        valmed = {}
        for m in minmed.keys():
            (l,v) = minmed[m]
            valmed[m] = v * scaler
        for j in range(len(list(varmed.keys()))):
            m = list(varmed.keys())[j]
            (l,v) = varmed[m]
            valmed[m] = X[i,j] * scaler
        if verbose == 2:
            print(f'start create_medium_run_cobra i={i}/{L}')
        try: 
            V[i], obj[i] = run_cobra(model, objective, valmed, method='FBA', 
                                     genekos=genekos, verbose=verbose)
        except:
            obj[i] = 0
            V[i] = {r.id: 0 for r in model.reactions}
        if verbose == 2:
            print(f'done create_medium_run_cobra i={i}/{L}')
            
    obj = np.asarray(list(obj.values()))
    V = np.asarray([list(V[i].values()) for i in range(L)])
    
    return obj, V

###############################################################################
# Creating, saving and loading training set object
# Training set object used in all modules
###############################################################################

def read_trainingfile(filename):
    # Read a training file
    # Returns:
    # - medium
    # - X and Y values
    from Library.Utilities import read_csv
    if not os.path.isfile(filename+'.csv'):
        sys.exit(f'{filename} file not found')    
    H, D = read_csv(filename)
    med, level, value = H, D[0], D[1]
    medium = {}
    for i in range(len(med)):
        if level[i] >= 0:
            medium[med[i]] = (level[i], value[i])
    minsize = list(level).count(1)
    medsize = len(medium.keys())
    X = D[2:,minsize:medsize]
    Y = D[2:,medsize:]
    return medium, X, Y

def write_trainingfile(filename, medium, X, Y):
    # Write a training file
    # Inputs:
    # - medium
    # - X and Y values
    
    H = list(medium.keys())
    L, V, minmed = {}, {}, 0
    for i in range(len(H)):
        (L[i], V[i]) = medium[H[i]]
        minmed = minmed+1 if L[i] == 1 else minmed
    L, V = np.asarray(list(L.values())), np.asarray(list(V.values()))
    D = np.concatenate((L, V), axis=0).reshape(2, L.shape[0])
    ones = np.ones((X.shape[0], minmed))
    X = np.concatenate((ones, X), axis=1)
    D = np.concatenate((D, X), axis=0)
    H.append('Y')
    neg = -np.ones((2, Y.shape[1]))
    Y = np.concatenate((neg, Y), axis=0)
    D = np.concatenate((D, Y), axis=1) 
    write_csv(filename, H, D)

class TrainingSet:
    # All elements necessary to run AMN
    # cf. save for definition of parameters
    def __init__(self,
                 cobraname='',
                 mediumname='',
                 method='FBA',
                 ratmed=0,
                 objective=[],
                 measure=[],
                 verbose=False):
        if cobraname == '':
            return  # create an empty object
        if not os.path.isfile(cobraname + '.xml'):
            sys.exit(f'{cobraname} cobra file not found')
        self.cobraname = cobraname  # model cobra file
        self.mediumname = mediumname  # medium file
        self.model = cobra.io.read_sbml_model(cobraname + '.xml')
        self.reduce = 0
        self.ratmed = ratmed
        self.method = method            
        self.medium, X, Y = read_trainingfile(mediumname)
        self.X, self.Y = X, Y
        self.size = Y.shape[0]
        if verbose:
            print(f'medium: {self.medium}')
            print(f'len medium: {len(self.medium.keys())}')
            print(f'ratmed: {self.ratmed}')
            print(f'method: {self.method}')
            print(f'X: {self.X.shape}')
            print(f'Y: {self.Y.shape}')
        # set objective and measured reactions lists
        self.objective = [get_objective(self.model)] \
            if objective == [] else objective
        self.measure = [r.id for r in self.model.reactions] \
            if measure == [] else measure
        if verbose:
            print(f'objective: {self.objective}')
            print(f'measurements size: {len(self.measure)}')
        # compute matrices and objective vector for AMN
        minmed, varmed, ko = get_minmed_varmed_ko(self.medium)
        self.S, self.Pin, self.Pko, self.Pout = \
            get_matrices(self.model, varmed, ko, self.measure,
                          self.model.reactions)
        # Import experimenal media from media file when X values are given
        # Reset X, Y and size
        if 'FBA' in self.method and self.X.shape[0]:
            self.mediumexp = self.import_mediumexp(self)
        else:
            self.mediumexp = []

    def reduce_model(self, eps=1e-3, verbose=False):
        # reduce a model removing zero fluxes
        # all fluxes < eps are set to zero
        
        # Collect reaction to be removed
        remove, model, flux = {}, self.model, self.Y
        flux[flux < eps] = 0
        for i in range(flux.shape[1]):
            if np.count_nonzero(flux[:, i]) == 0 and \
                model.reactions[i].id not in self.medium.keys():
                remove[i] = model.reactions[i]
                if verbose:
                    print(f'delete reaction: {model.reactions[i].id}')

        # Actual deletion
        model.remove_reactions(list(remove.values()))
        manip.delete.prune_unused_reactions(model)
        for m in model.metabolites:
            if len(m.reactions) == 0:
                if verbose:
                    print(f'delete metabolite: {m.id}')
                model.remove_metabolites(m)
        manip.delete.prune_unused_metabolites(model)
        print(f'reduced numbers of metabolites and reactions: '
              f'{len(model.metabolites)}, {len(model.reactions)}')
        
        # update model
        zero_columns = np.argwhere(np.all(flux == 0, axis=0)).flatten()
        self.Y = np.delete(flux, zero_columns, axis=1)       
        self.model = model
        self.measure = [r.id for r in self.model.reactions]

    def save(self, filename, reduce=0, verbose=False):
        # save cobra model in xml and parameter in npz (compressed npy)
        self.reduce = reduce
        if self.reduce:
            self.reduce_model(eps=reduce, verbose=verbose)
        # Recompute matrices
        minmed, varmed, ko = get_minmed_varmed_ko(self.medium)
        self.S, self.Pin, self.Pko, self.Pout = \
        get_matrices(self.model, varmed, ko, self.measure,
                         self.model.reactions)
        # save cobra file
        cobra.io.write_sbml_model(self.model, filename+'.xml')
        # save parameters
        nammed, levmed, valmed = np.asarray(list(self.medium.keys())), {}, {}
        for i in range(len(nammed)):
            m = nammed[i]
            (levmed[i], valmed[i]) = self.medium[m]
        levmed = np.asarray(list(levmed.values()))
        valmed = np.asarray(list(valmed.values())) 
        np.savez_compressed(filename, 
                            cobraname = filename,
                            reduce = self.reduce,
                            mediumname = self.mediumname,
                            nammed = nammed, # to save the medium dic
                            levmed = levmed,
                            valmed = valmed,
                            mediumexp = self.mediumexp,
                            objective = self.objective,
                            method = self.method,
                            size = self.size,
                            measure = self.measure,
                            S = self.S,
                            Pin = self.Pin,
                            Pko = self.Pko,
                            Pout = self.Pout,
                            X = self.X, 
                            Y = self.Y)
        
    def load(self, filename):
        # load parameters (npz format)
        if not os.path.isfile(filename+'.npz'):
            print(filename+'.npz')
            sys.exit('file not found')
        loaded = np.load(filename+'.npz', allow_pickle=True)
        self.cobraname = str(loaded['cobraname'])
        self.reduce = loaded['reduce']
        self.mediumname = str(loaded['mediumname'])
        nammed = loaded['nammed']
        levmed = loaded['levmed']
        valmed = loaded['valmed']
        self.medium = {}
        for i in range(len(nammed)):
            m = nammed[i]
            self.medium[m] = (levmed[i], valmed[i])
        self.mediumexp = loaded['mediumexp']
        self.objective = loaded['objective']
        self.method = str(loaded['method'])
        self.size = loaded['size']
        self.measure = loaded['measure']
        self.S = loaded['S']
        self.Pin = loaded['Pin']
        if 'Pko' in loaded.keys():
            self.Pko = loaded['Pko']
        else:
            self.Pko = np.asarray([])
        self.Pout = loaded['Pout']
        self.X = loaded['X']
        self.Y = loaded['Y']
        self.model = cobra.io.read_sbml_model(self.cobraname+'.xml')

    def printout(self,filename=''):
        if filename != '':
            sys.stdout = open(filename, 'wb')
        print(f'model file name: {self.cobraname}')
        print(f'reduced model (min flux): {self.reduce}')
        print(f'medium file name: {self.mediumname}')
        print(f'medium: {len(self.medium)}')
        print(f'experimental medium: {len(self.mediumexp)}')
        print(f'list of reactions in objective: {self.objective}')
        print(f'method: {self.method}')
        print(f'trainingsize: {self.size}')
        print(f'list of measured reactions: {len(self.measure)}')
        print(f'Stoichiometric matrix: {self.S.shape}')
        print(f'Boundary matrix from reactions to medium: {self.Pin.shape}')
        print(f'KO matrix from reactions to ko: {self.Pko.shape}')
        print(f'Measurement matrix from reaction to measures: {self.Pout.shape}')
        print(f'Training set X: {self.X.shape}')
        print(f'Training set Y: {self.Y.shape}')
        if filename != '':
            sys.stdout.close()

    def get(self, sample_size=100, mediumexp=[], genekos=[], 
            cobra_min_objective=1.0e-3,
            cobra_max_objective=3.0,
            reduce=0, verbose=False):
        # Generate a training set for AMN
        # Inputs: 
        # - sample size
        # - input medium list
        # Returns: 
        # - X, Y : medium and reaction flux values after running Cobra

        X, Y, valmed, I = {}, {}, {}, 0
        while I < sample_size:
            x, y = get_io_cobra(self.model, self.objective,
                                self.medium, mediumexp, self.ratmed,
                                valmed=valmed,
                                method=self.method, genekos=genekos,
                                cobra_min_objective=1.0e-3,
                                cobra_max_objective=3.0,
                                verbose=verbose)
            if I % 100 == 0:
                print(f'-----------------------------{self.cobraname} sample: {I} -----------------------------')
            if np.sum(y): # only non null solutions
                X[I], Y[I] = x, y
                I += 1
        X = np.asarray(list(X.values()))
        Y = np.asarray(list(Y.values()))

        # In case 'get' is called several times
        if self.X.shape[0] > 0 and reduce == 0:
            self.X = np.concatenate((self.X, X), axis=0)
            self.Y = np.concatenate((self.Y, Y), axis=0)
        else:
            self.X, self.Y = X, Y
        self.size = self.X.shape[0]

    def import_mediumexp(self, reset=True):
        # Input:
        # - reset: when reset reset to zero parameter X, Y
        # Return:
        # - import_medium: the list of variable medium based on self.X

        mediumexp = {}
        minmed, varmed, ko = get_minmed_varmed_ko(self.medium)
        X, Y = self.X, self.Y
        for i in range(X.shape[0]):
            mediumexp[i] = []
            for j in range(X.shape[1]):
                m = list(varmed.keys())[j]
                if X[i, j] > 0:
                    mediumexp[i].append(m)
        mediumexp = list(mediumexp.values())
        return np.asarray(mediumexp, dtype=object)

    def filter_measure(self, measure=[], verbose=False):
        # Keep only reaction fluxes in measure
        # Input:
        # - measure: a list of measured reaction fluxes
        # Output:
        # - self.Yall all reactions
        
        self.measure = measure if len(self.measure) > 0 else self.measure
        minmed, varmed, ko = get_minmed_varmed_ko(self.medium)
        _, _, _, self.Pout = \
        get_matrices(self.model, varmed, ko, self.measure, self.model.reactions)
        self.Yall = self.Y.copy()
        if len(self.measure) > 0:
            # Y = only the reaction fluxes that are in Vout
            Y = np.matmul(self.Y,np.transpose(self.Pout)) \
            if ('EXP') not in self.method else self.Y
            self.Y = Y
        if verbose:
            print(f'number of reactions: {self.S.shape[1]}={self.Yall.shape[1]}')
            print(f'number of metabolites: {self.S.shape[0]}')
            print(f'filtered measurements size: {self.Y.shape[1]}')

    def filter_medium(self, verbose=False):
        # Keep only medium that are exchange reaction in model

        medium = {}
        exchange = [r.id for r in self.model.reactions if '_e' in r.id]
        for rid in self.medium.keys():
            if rid in exchange:
                medium[rid] = self.medium[rid]  
        self.medium = medium
        if verbose:
            print(f'reduced medium to echange reaction: {self.medium}')
