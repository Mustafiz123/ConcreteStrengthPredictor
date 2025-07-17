# --- Feature Scaling Normalise given data

def normalize_features(X):
    """
    zscore normalisation:

    Parameters
    ----------
    X         : Numpy array, training data.

    Returns
    -------
    X_normal  : Nomrlaised X array.
    mean      : Mean of each colun or feature.
    std       : standard deviation of each column or feature.
    """
    # Mean of each column
    mean     = np.mean(X, axis=0)  # axis=0, refers to column wise mean evaluation
    # Standard deviation of each column
    std  = np.std(X, axis=0)       # axis=0, refers to column wise mean evaluation
    # z score evaluation (this is a element wise operation, also known as broadcasting)
    X_normal = (X - mean) / std      # X_normal is the z-score

    return (X_normal, mean, std)

# --- Evaluate and return the total cost 

import numpy as np

def eval_cost(x, y, w, b):
    """
    Total cost evaluation:

    Parameters
    ----------
    x       : Numpy array, training data.
    y       : Target variable, training data
    w       : Array of weights
    b       : Bias

    Returns
    -------
    cost    : Total Cost
    """
    x_nrows = x.shape[0]
    x_ncols = x.shape[1]
    cost = 0
    for i in range(0, x_nrows):
        y_predict = np.dot(x[i,:],w) + b
        cost = cost + (1/(2*x_nrows))*(y_predict - y[i])**2
    
    return cost

# --- Evaluate and return the gradients

def eval_gradient(x, y, w, b):
    """
    Gradient evaluation:

    Parameters
    ----------
    x       : Numpy array, training data.
    y       : Target variable, training data
    w       : Array of weights
    b       : Bias

    Returns
    -------
    dJdw   : Array of gradients for weights
    dJdb   : Gradient for bias
    """    
    x_rows = x.shape[0]
    x_cols = x.shape[1]
    dJdw = np.zeros(x_cols)
    dJdb = 0.0
    
    for i in range(x_rows):
        error = (np.dot(x[i], w) + b) - y[i]
        for j in range(x_cols):
            dJdw[j] = dJdw[j] + error*x[i,j]
        dJdb = dJdb + error
        
    dJdw = dJdw/x_rows
    dJdb = dJdb/x_rows
    
    return dJdw, dJdb

