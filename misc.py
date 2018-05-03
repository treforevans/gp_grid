# utilities for testing
import numpy as np

def rastrigin(x,lin_term=None):
    """
    Rastrigin test function

    input should be in range [-5.12, 5.12]
    if x in range [0,1], can transform by rastrigin((x*2-1)*5.12)

    if lin_term is not None then will add a linear term to the first dimension. This helps
    to make the function non-symetric wrt the input dimensions
    """
    assert x.ndim == 2
    d = x.shape[1]
    f = 10*d
    for i in range(d):
        f = f+(np.power(x[:,i,None],2) - 10*np.cos(2*np.pi*x[:,i,None]));
    if lin_term is not None:
        f += lin_term*x[:,(0,)]
    return f


def grid2mat(*xg):
    """
    transforms a grid to a numpy matrix of points

    Inputs:
        *xg : ith input is a 1D array of the grid locations along the ith dimension

    Outputs:
        x : numpy matrix of size (N,d) where N = n1*n2*...*nd = len(xg[0])*len(xg[1])*...*len(xg[d-1])
    """
    X_mesh = nd_grid(*xg) # this is the meshgrid, all I have to do is flatten it
    d = X_mesh.shape[0]
    N = X_mesh[0].size
    x = np.zeros((N, d)) # initialize
    for i, X1d in enumerate(X_mesh): # for each 1d component of the mesh
        x[:,i] = X1d.reshape(-1, order='C') # reshape it into a vector
    return x


def nd_grid(*xg):
    """
    This mimics the behaviour of nd_grid in matlab.
    (np.mgrid behaves similarly however I don't get how to call it clearly.)
    """
    grid_shape = [np.shape(xg1d)[0] for xg1d in xg] # shape of the grid
    d = np.size(grid_shape)
    N = np.product(grid_shape)
    X_mesh = np.empty(d, dtype=object)
    for i, xg1d in enumerate(xg): # for each 1d component
        if np.ndim(xg1d) > 1:
            assert np.shape(xg1d)[1] == 1, "only currently support each grid dimension being 1d"
        n = np.shape(xg1d)[0] # number of points along dimension of grid
        slice_shape = np.ones(d, dtype=int);   slice_shape[i] = n # shape of the slice where xg1d fits
        stack_shape = np.copy(grid_shape);     stack_shape[i] = 1 # shape of how the slice should be tiled
        X_mesh[i] = np.tile(xg1d.reshape(slice_shape), stack_shape) # this is the single dimension on the full grid
    return X_mesh


def gapify_data(y, gappiness):
    """apply gaps to data"""
    if gappiness < 1: # then it is a fraction
        N = y.size
        N_missing = np.int32(np.floor(gappiness*N))
    else: # is the number of missing points
        N_missing = gappiness
    gaps = np.random.choice(y.shape[0], size=N_missing, replace=False) # randomly set some data to nan
    y_gappy = y.copy()
    y_gappy[gaps] = np.nan
    return gaps,y_gappy


def elastic_membrane(x,y,t,F,H,C,D,lx=1,ly=1,v=1):
    """
    2d elastic membrane solution constrained to zero displacement on the boundary within
    the range:
        x in [0,lx]
        y in [0,ly]

    Inputs:
        x,y,t : spacial-time points. all of size (n,1)
        F : (n_p,1) vector of coeffiencts
        H : (n_q,1) vector of coefficients
        C,D : coefficients of each frequency. This is a matrix
            of size (n_p,n_q) where:
                n_p is number of wave components along x
                n_q is number of wave components along y
        v : velocity of wave

    Outputs:
        z: vertical displacement of membrane. Array of size (n,1)
    """
    # check the input points
    assert x.shape[1] == 1
    assert y.shape[1] == 1
    assert t.shape[1] == 1
    assert x.shape[0] == y.shape[0] == t.shape[0]

    # get the number of wave numbers to consider and error check sizes
    assert F.shape[1] == 1
    assert H.shape[1] == 1
    n_p = F.shape[0]
    n_q = H.shape[0]
    p = np.tile(np.arange(1,n_p+1).reshape((-1,1)),(1,n_q))
    q = np.tile(np.arange(1,n_q+1).reshape((1,-1)),(n_p,1))
    assert C.shape == (n_p,n_q)
    assert D.shape == (n_p,n_q)

    # determine the frequencies
    w = np.pi*v*np.sqrt(np.power(p/lx,2) + np.power(q/ly,2))

    # determine the coefficients
    A = C * F.dot(H.T)
    B = D * F.dot(H.T)

    # compute the membrane displacement
    x = np.tile(x.reshape((1,1,-1)),(n_p,n_q,1))
    y = np.tile(y.reshape((1,1,-1)),(n_p,n_q,1))
    t = np.tile(t.reshape((1,1,-1)),(n_p,n_q,1))
    A = A.reshape(A.shape + (1,))
    B = B.reshape(B.shape + (1,))
    p = p.reshape(p.shape + (1,))
    q = q.reshape(q.shape + (1,))
    w = w.reshape(w.shape + (1,))
    z = np.sum(
        (A*np.cos(w*t)+B*np.sin(w*t))*np.sin(p*np.pi*x/lx)*np.sin(q*np.pi*y/ly),
        axis=(0,1)).reshape((-1,1))
    return z


