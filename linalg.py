import numpy as np
import logging
import warnings
logger = logging.getLogger(__name__)
from pdb import set_trace
import sys


def log_kron(a, b, a_logged=False, b_logged=False):
    """
    computes np.log(np.kron(a,b)) in a numerically stable fashion which
    decreases floating point error issues especially when many kron products
    are evaluated sequentially

    Inputs:
        a_logged : specify True if the logarithm of a has already been taken
        b_logged : specify True if the logarithm of b has already been taken
    """
    assert a.ndim == b.ndim == 1, "currenly only working for 1d arrays"
    if not a_logged:
        a = np.log(a)
    if not b_logged:
        b = np.log(b)
    return (a.reshape((-1,1)) + b.reshape((1,-1))).reshape(-1)


class solver_counter:
    """
    counter for pcg, gmres, ... scipy routines since they don't keep count of iterations.
    """

    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.backup= None


    def __call__(self, rk=None, msg='', store=None):
        self.niter += 1
        if self._disp:
            logger.info('iter %3i. %s' % (self.niter,msg))
            sys.stdout.flush()
        if store is not None: # then backup the value
            self.backup = store


class LogexpTransformation:
    """ apply log transformation to positive parameters for optimization """
    _lim_val = 36.
    _log_lim_val = np.log(np.finfo(np.float64).max)


    def inverse_transform(self, x):
        return np.where(x > self._lim_val, x, np.log1p(np.exp(np.clip(x, -self._log_lim_val, self._lim_val))))


    def transform(self, f):
        return np.where(f > self._lim_val, f, np.log(np.expm1(f)))


    def transform_grad(self, f, grad_f):
        return grad_f*np.where(f > self._lim_val, 1.,  - np.expm1(-f))


