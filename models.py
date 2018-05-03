from .kern import BaseKernel, GridKernel
from .tensors import SelectionMatrix
from .linalg import solver_counter, LogexpTransformation

# numpy/scipy stuff
import numpy as np
from scipy.sparse.linalg import cg as pcg
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import lu_factor,lu_solve,cho_factor,cho_solve
from scipy.optimize import fmin_l_bfgs_b

# development stuff
from numpy.linalg.linalg import LinAlgError
from numpy.testing import assert_array_almost_equal
from traceback import format_exc
from pdb import set_trace
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)

class BaseModel(object):
    param_shift = {'+ve':1e-200} # for positive constrained problems, this is as close as it can get to zero, by default this is not used
    _transformations = {'+ve':LogexpTransformation()}

    def __init__(self):
        """ initialize a few instance variables """
        logger.debug('Initializing %s model.' % self.__class__.__name__)
        self.dependent_attributes = ['_alpha','_log_like','_gradient','_K','_log_det'] # these are set to None whenever parameters are updated
        self._previous_parameters = None # previous parameters from last call to parameters property
        self.grad_method = None # could be {'finite_difference','adjoint'}
        self.noise_var_constraint = '+ve' # constraint for the Gaussian noise variance
        return


    def log_likelihood(self, return_gradient=False):
        """
        computes the log likelihood and the gradient (if hasn't already been computed).

        If the return_gradient flag is set then then the gradient will be returned as a second arguement.
        """
        p = self.parameters # this has to be called first to fetch parameters and ensure internal consistency

        # check if I need to recompute anything
        if return_gradient and (self._gradient is None):
            # compute the log likelihood and gradient wrt the parameters
            if 'adjoint' in self.grad_method:
                (self._log_like, self._gradient) = self._adjoint_gradient(p)
            elif 'finite_difference' in self.grad_method:
                (self._log_like, self._gradient) = self._finite_diff_gradient(p)
            else:
                raise RuntimeError('unknown grad_method %s' % repr(self.grad_method))
        elif self._log_like is None: # just compute the log-likelihood without the gradient
            self._log_like = self._compute_log_likelihood(p)
        else: # everything is already computed
            pass

        if return_gradient: # return both
            return self._log_like, self._gradient
        else: # just return likelihood
            return self._log_like


    def optimize(self, max_iters=1000, messages=True, use_counter=False, factr=10000000.0, pgtol=1e-05):
        """
        maximize the log likelihood

        Inputs:
            max_iters : int
                maximum number of optimization iterations
            factr, pgtol : lbfgsb convergence criteria, see fmin_l_bfgs_b help for more details
                use factr of 1e12 for low accuracy, 10 for extremely high accuracy (default 1e7)
        """
        logger.debug('Beginning MLE to optimize hyperparameters. grad_method=%s' % self.grad_method)

        # setup the optimization
        try:
            x0 = self._transform_parameters(self.parameters) # get the transformed value to start at
            assert np.all(np.isfinite(x0)), "initial transformation led to non-finite values"
        except:
            logger.error('Transformation failed for initial values. Ensure constraints are met or the value is not too small.')
            raise

        # filter out the fixed parameters
        free = np.logical_not(self._fixed_indicies)
        x0 = x0[free]

        # setup the counter
        if use_counter:
            self._counter = solver_counter(disp=True)
        else:
            self._counter = None

        # run the optimization
        try:
            x_opt, f_opt, opt = fmin_l_bfgs_b(func=self._objective_grad, x0=x0, factr=factr, pgtol=pgtol, maxiter=max_iters, disp=messages)
        except (KeyboardInterrupt,IndexError): # sometimes interrupting gives index error for scipy sparse matricies it seems
            logger.info('Keyboard interrupt raised. Cleaning up...')
            if self._counter is not None and self._counter.backup is not None:# use the backed up copy of parameters from the last iteration
                self.parameters = self._counter.backup[1]
                logger.info('will return best parameter set with log-likelihood = %.4g' % self._counter.backup[0])
        else:
            logger.info('Function Evals: %d. Exit status: %s' % (f_opt, opt['warnflag']))
            # extract the optimal value and set the parameters to this
            transformed_parameters = self._previous_parameters # the default parameters are the previous ones
            transformed_parameters[free] = x_opt # these are the transformed optimal parameters
            self.parameters = self._untransform_parameters(transformed_parameters) # untransform
        return opt


    def checkgrad(self, decimal=3, raise_if_fails=True):
        """
        checks the gradient and raises error if does not pass
        """
        grad_exact = self._finite_diff_gradient(self.parameters)[1]
        grad_exact[self._fixed_indicies] = 1 # I don't care about the gradients of fixed variables
        grad_analytic = self.log_likelihood(return_gradient=True)[1]
        grad_analytic[self._fixed_indicies] = 1 # I don't care about the gradients of fixed variables

        # first protect from nan values incase both analytic and exact are small
        protected_nan = np.logical_and(np.abs(grad_exact) < 1e-8, np.abs(grad_analytic) < 1e-8)

        # now protect against division by zero. I do this by just removing the values that have a small absoute error
        # since if the absolute error is tiny then I don't really care about relative error
        protected_div0 = np.abs(grad_exact-grad_analytic) < 1e-5

        # now artificially change these protected values
        grad_exact[np.logical_or(protected_nan, protected_div0)] = 1.
        grad_analytic[np.logical_or(protected_nan, protected_div0)] = 1.

        try:
            assert_array_almost_equal(grad_exact / grad_analytic, np.ones(grad_exact.shape),
                                      decimal=decimal, err_msg='Gradient ratio did not meet tolerance.')
        except:
            logger.info('Gradient check failed.')
            logger.debug('[[Finite-Difference Gradient], [Analytical Gradient]]:\n%s\n' % repr(np.asarray([grad_exact,grad_analytic])))
            if raise_if_fails:
                raise
            else:
                logger.info(format_exc()) # print the output
                return False
        else:
            logger.info('Gradient check passed.')
            return True

    @property
    def parameters(self):
        """
        this gets the parameters from the object attributes
        """
        parameters = np.concatenate((np.ravel(self.noise_var), self.kern.parameters),axis=0)

        # check if the parameters have changed
        if not np.array_equal(parameters, self._previous_parameters):
            # remove the internal variables that rely on the parameters
            for attr in self.dependent_attributes:
                setattr(self, attr, None)
            # update the previous parameter array
            self._previous_parameters = parameters.copy()
        return parameters.copy()

    @parameters.setter
    def parameters(self,parameters):
        """
        this takes the optimization variables parameters and sets the internal state of self
        to make it consistent with the variables
        """
        # set the parameters internally
        self.noise_var       = parameters[0]
        self.kern.parameters = parameters[1:]

        # check if the parameters have changed
        if not np.array_equal(parameters, self._previous_parameters):
            # remove the internal variables that rely on the parameters
            for attr in self.dependent_attributes:
                setattr(self, attr, None)
            # update the previous parameter array
            self._previous_parameters = parameters.copy()
        return parameters

    @property
    def constraints(self):
        """ returns the model parameter constraints as a list """
        constraints = np.concatenate((np.ravel(self.noise_var_constraint), self.kern.constraints),axis=0)
        return constraints


    def predict(self,Xnew,compute_var=None):
        """
        make predictions at new points

        MUST begin with a call to parameters property to ensure internal state consistent
        """
        raise NotImplementedError('')


    def fit(self):
        """
        determines the weight vector _alpha

        MUST begin with a call to parameters property to ensure internal state consistent
        """
        raise NotImplementedError('')


    def _objective_grad(self,transformed_free_parameters):
        """ determines the objective and gradients in the transformed input space """
        # get the fixed indices and add to the transformed parameters
        free = np.logical_not(self._fixed_indicies)
        transformed_parameters = self._previous_parameters # the default parameters are the previous ones
        transformed_parameters[free] = transformed_free_parameters
        try:
            # untransform and internalize parameters
            self.parameters = self._untransform_parameters(transformed_parameters)
            # compute objective and gradient in untransformed space
            (objective, gradient) = self.log_likelihood(return_gradient=True)
            objective = -objective # since we want to minimize
            gradient =  -gradient
            # ensure the values are finite
            if not np.isfinite(objective):
                logger.debug('objective is not finite')
            if not np.all(np.isfinite(gradient[free])):
                logger.debug('some derivatives are non-finite')
            # transform the gradient 
            gradient = self._transform_gradient(self.parameters, gradient)
        except (LinAlgError, ZeroDivisionError, ValueError):
            logger.error('numerical issue while computing the log-likelihood or gradient.')
            logger.debug('Here is the current model where the failure occured:\n' + self.__str__())
            raise
        # get rid of the gradients of the fixed parameters
        free_gradient = gradient[free]

        # call the counter if ness
        if self._counter is not None:
            msg='log-likelihood=%.4g, gradient_norm=%.2g' % (-objective, np.linalg.norm(gradient))
            if self._counter.backup is None or self._counter.backup[0] < -objective: # then update backup
                self._counter(msg=msg,store=(-objective,self.parameters.copy()))
            else: # don't update backup
                self._counter(msg=msg)
        return objective, free_gradient

    @property
    def _fixed_indicies(self):
        """ returns a bool array specifiying where the indicies are fixed """
        fixed_inds = self.constraints == 'fixed'
        return fixed_inds

    @property
    def _free_indicies(self):
        """ returns a bool array specifiying where the indicies are free """
        return np.logical_not(self._fixed_indicies)


    def _transform_parameters(self, parameters):
        """
        applies a transformation to the parameters based on a constraint
        """
        constraints = self.constraints
        assert parameters.size == np.size(constraints) # check if sizes correct
        transformed_parameters = np.zeros(parameters.size)
        for i,(param,constraint) in enumerate(zip(parameters,constraints)):
            if constraint is None or constraint == 'fixed' or constraint == '': # then no transformation
                transformed_parameters[i] = param
            else: # I need to transform the parameters
                transformed_parameters[i] = self._transformations[constraint].transform(param - self.param_shift[constraint])

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(transformed_parameters)):
            logger.debug('transformation led to non-finite value')
        return transformed_parameters


    def _transform_gradient(self, parameters, gradients):
        """
        see _transform parameters
        """
        constraints = self.constraints
        assert parameters.size == gradients.size == np.size(constraints) # check if sizes correct
        transformed_grads      = np.zeros(parameters.size)
        for i,(param,grad,constraint) in enumerate(zip(parameters,gradients,constraints)):
            if constraint is None or constraint == '': # then no transformation
                transformed_grads[i] = grad
            elif constraint != 'fixed': # then apply a transformation (if fixed then do nothing)
                transformed_grads[i] = self._transformations[constraint].transform_grad(param - self.param_shift[constraint],grad)

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(transformed_grads)):
            logger.debug('transformation led to non-finite value')
        return transformed_grads


    def _untransform_parameters(self, transformed_parameters):
        """ applies a reverse transformation to the parameters given constraints"""
        assert transformed_parameters.size == np.size(self.constraints) # check if sizes correct
        parameters = np.zeros(transformed_parameters.size)
        for i,(t_param,constraint) in enumerate(zip(transformed_parameters,self.constraints)):
            if constraint is None or constraint == 'fixed' or constraint == '': # then no transformation
                parameters[i] = t_param
            else:
                parameters[i] = self._transformations[constraint].inverse_transform(t_param) + self.param_shift[constraint]

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(parameters)):
            logger.debug('transformation led to non-finite value')
        return parameters


    def _finite_diff_gradient(self, parameters):
        """
        helper function to compute function gradients by finite difference.

        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters
            log_like : float
                log likelihood at the current point

        Outputs:
            log_likelihood
        """
        assert isinstance(parameters,np.ndarray)
        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]

        # first take a forward step in each direction
        step = 1e-6 # finite difference step
        log_like_fs = np.zeros(free_inds.size)
        for i,param_idx in enumerate(free_inds): # loop through all free indicies
            p_fs = parameters.copy()
            p_fs[param_idx] += step # take a step forward
            log_like_fs[i] = self._compute_log_likelihood(p_fs) # compute the log likelihood at the forward step

        # compute the log likelihood at current point
        log_like = self._compute_log_likelihood(parameters)

        # compute the gradient
        gradient = np.zeros(parameters.shape) # default gradient is zero
        gradient[free_inds] = (log_like_fs-log_like) # compute the difference for the free parameters
        #if np.any(np.abs(gradient[free_inds]) < 1e-12):
            #logger.debug('difference computed during finite-difference step is too small. Results may be inaccurate.')
        gradient[free_inds] = gradient[free_inds]/step # divide by the step length
        return log_like, gradient


    def _compute_log_likelihood(self, parameters):
        """
        helper function to compute log likelihood.
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters

        Outputs:
            log_likelihood
        """
        raise NotImplementedError('')


    def _adjoint_gradient(self,parameters):
        raise NotImplementedError('')
        return log_like, gradient


    def __str__(self):
        from tabulate import tabulate
        s = '\n%s Model\n' % self.__class__.__name__

        # print the  noise_var stuff
        s += str(tabulate([['noise_var',self.noise_var,self.noise_var_constraint]],
                          headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl')) + '\n'

        # print the kernel stuff
        s += str(self.kern)
        return s


class GPRegression(BaseModel):
    """
    general GP regression model
    """

    def __init__(self, X, Y, kernel, noise_var=1.):
        # call init of the super method
        super(GPRegression, self).__init__()
        # check inputs
        assert X.ndim == 2
        assert Y.ndim == 2
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        assert not np.any(np.isnan(Y))
        self.num_data, self.input_dim = self.X.shape
        if Y.shape[0] != self.num_data:
            raise ValueError('X and Y sizes are inconsistent')
        self.output_dim = self.Y.shape[1]
        if self.output_dim != 1:
            raise RuntimeError('this only deals with 1 response for now')
        assert isinstance(kernel, BaseKernel)
        self.kern = kernel

        # add the noise_var internally
        self.noise_var = np.float64(noise_var)

        # set some defaults
        self.grad_method = 'finite_difference chol'
        return


    def fit(self):
        """ finds the weight vector alpha """
        logger.debug('Fitting; determining weight vector.')
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then need to find the new alpha
            self._alpha = np.linalg.solve(self.kern.cov(x=self.X) + self.noise_var*np.eye(self.num_data),
                                         self.Y)


    def predict(self,Xnew,compute_var=None):
        """
        make predictions at new points

        Inputs:
            Xnew : (M,d) numpy array of points to predict at
            compute_var : whether to compute the variance at the test points
                * None (default) : don't compute variance
                * 'diag' : return the diagonal of the covariance matrix, size (M,1)
                * 'full' : return the full covariance matrix of size (M,M)

        Outputs:
            Yhat : (M,1) numpy array predictions at Xnew
            Yhatvar : only returned if compute_var is not None. See `compute_var` input
                notes for details
        """
        logger.debug('Predicting model at new points.')
        assert Xnew.ndim == 2
        assert Xnew.shape[1] == self.input_dim
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then I need to train
            self.fit()

        # get cross covariance between training and testing points
        Khat = self.kern.cov(x=Xnew, z=self.X)

        # predict the mean at the test points
        Yhat = Khat.dot(self._alpha)

        # predict the variance at the test points
        # TODO: make this more efficient, especially for diagonal predictions
        if compute_var is not None:
            Yhatvar = self.kern.cov(x=Xnew) + self.noise_var*np.eye(Xnew.shape[0]) - \
                    Khat.dot(np.linalg.solve(self.kern.cov(x=self.X) + self.noise_var*np.eye(self.num_data),
                                               Khat.T))
            if compute_var == 'diag':
                Yhatvar = Yhatvar.diag().reshape((-1,1))
            elif compute_var != 'full':
                raise ValueError('Unknown compute_var = %s' % repr(compute_var))
            return Yhat,Yhatvar
        else: # just return the mean
            return Yhat


    def _compute_log_likelihood(self, parameters):
        """
        helper function to compute log likelihood
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters

        Outputs:
            log_likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # compute the new covariance
        K = self.kern.cov(self.X)

        # compute the log likelihood
        if 'svd' in self.grad_method: # then compute using svd
            (Q,eig_vals) = np.linalg.svd(K, full_matrices=0, compute_uv=1)[:2]
            log_like = -0.5*np.sum(np.log(eig_vals+self.noise_var)) - \
                        0.5*np.dot(self.Y.T, solve_schur(Q,eig_vals,self.Y,shift=self.noise_var)) - \
                        0.5*self.num_data*np.log(np.pi*2)
        if 'chol' in self.grad_method: # then compute using cholesky factorization
            U = cho_factor(K + self.noise_var * np.eye(self.num_data))
            log_like = -np.sum(np.log(np.diagonal(U[0],offset=0,axis1=-1, axis2=-2))) - \
                        0.5*np.dot(self.Y.T, cho_solve(U,self.Y)) - \
                        0.5*self.num_data*np.log(np.pi*2)
        else: # just use logpdf from scipy 
            log_like = mvn.logpdf(self.Y.squeeze(),
                                  np.zeros(self.num_data),
                                  K + self.noise_var * np.eye(self.num_data))
        return log_like


class GPGridRegression(BaseModel):
    """
    GP regression model for points on a grid
    """

    def __init__(self, Xg, Yg, kern_list, noise_var=1.):
        """
        Inputs:
            Xg : list of length d
                each element in X must be the array of points along each dimension of the grid. eg.
                    X[i].shape = (grid_shape[i], grid_sub_dim[i])
            Yg : nd_array of responses. eg.
                    Yg.shape = grid_shape
            kern_list : list of kerels for each grid dimension.
                Each sub-kernel must have approriate dimensions:
                    kernel[i].n_dims = d_grid[i]
                Ultimately these kernels will be multiplied together.
            noise_var : see models.Regression
        """
        # call init of the super method
        super(GPGridRegression, self).__init__()
        # first make all inputs a numpy array
        Xg        = np.asarray(Xg)
        Yg        = np.asarray(Yg)
        # check X
        self.grid_dim = Xg.shape[0] # number of grid dimensions
        self.grid_shape   = np.zeros(self.grid_dim, dtype=int) # number of points along each sub dimension
        self.grid_sub_dim = np.zeros(self.grid_dim, dtype=int) # number of sub dimensions along each grid dim
        for i,X in enumerate(Xg): # loop over grid dimensions
            assert X.ndim == 2, "each element in Xg must be a 2d array"
            self.grid_sub_dim[i] = X.shape[1]
            self.grid_shape[i]   = X.shape[0]
        self.input_dim = np.sum(self.grid_sub_dim) # total number of dimensions
        self.num_data = np.prod(self.grid_shape) # total number of points on the full grid
        self.Xg = Xg.copy()
        # check Yg
        assert np.all(Yg.shape == self.grid_shape), "ensure response shape is same as grid shape"
        self.Y = Yg.reshape((-1,1), order='F') # reshape Y to a vector, Yg will recover shape
        # check the kernel
        assert isinstance(kern_list, (list,tuple,np.ndarray))
        assert np.ndim(kern_list) == 1, "kern_list can only be a 1d array of objects"
        for i,ki in enumerate(kern_list):
            assert isinstance(ki, BaseKernel), "ensure kern_list is a list of kernels"
            assert ki.n_dims == self.grid_sub_dim[i], "ensure the kernels in kern_list are the correct dimensionlity"
        self.kern = GridKernel(kern_list) # create kernel wrapper object

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # set some defaults
        self.grad_method = 'adjoint'
        self.dim_noise_var = 1e-12 # value to add to the diagonal of each dimensions covariance for conditioning

        # add to dependent attributes
        self.dependent_attributes = np.unique(np.concatenate(
            (self.dependent_attributes,
             [
              '_log_det', # stuff for the marginal likelihood
              '_Q','_Tp','_big_eig_vals','_A_decomp','_S', # cov rank reduced stuff
              '_Abar_decomp','_Sbar', '_zeta','_small_eig_vals', # FG preconditioner stuff
             ])))
        return

    @property
    def Yg(self):
        """ return the original 'grid' shape of y """
        return self.Y.reshape(self.grid_shape, order='F')


    def _compute_log_likelihood(self, parameters):
        """
        compute log likelihood
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters

        Outputs:
            log_likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # get the covariance matrix
        K = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var)

        # compute svd
        (Q,eig_vals) = K.svd()
        eig_vals = eig_vals.expand()
        self._log_det = np.sum(np.log(eig_vals+self.noise_var))
        self._alpha = Q.solve_schur(eig_vals,self.Y,shift=self.noise_var)
        log_like = -0.5*self._log_det - \
                    0.5*np.dot(self.Y.T, self._alpha) - \
                    0.5*self.num_data*np.log(np.pi*2)
        return log_like.squeeze()


    def fit(self):
        """ finds the weight vector alpha """
        logger.debug('Fitting; determining weight vector.')
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then need to find the new alpha
            # get the covariance matrix
            K = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var)

            # compute svd
            (Q,eig_vals) = K.svd()
            eig_vals = eig_vals.expand()

            # compute by solving a linear system of equations
            self._alpha = Q.solve_schur(eig_vals,self.Y,shift=self.noise_var)


    def predict_grid(self,Xg_new,compute_var=None):
        """
        make predictions at new points

        Inputs:
            Xg_new : list of length d
                each element in X must be the array of points along each dimension of the grid. eg.
                    X[i].shape = (grid_shape[i], grid_sub_dim[i])
            compute_var : whether to compute the variance at the test points
                * None (default) : don't compute variance
                * 'diag' : return the diagonal of the covariance matrix, size (M,1)
                * 'full' : return the full covariance matrix of size (M,M)

        Outputs:
            Yg_hat : numpy array of predictions of shape grid_shape at Xg_new
                to convert to a vector do:
                    Yg_hat.reshape((-1,1), order='F')
                which will make the vector consistent with the posterior covariance matrix
            Yhatvar : only returned if compute_var is not None. See `compute_var` input
                notes for details
                Note that this matrix corresponds with the vector form of Yg_hat (see above)
        """
        logger.debug('Predicting model at new points.')
        Xg_new = np.asarray(Xg_new)

        # check the format of the testing points
        assert self.grid_dim == Xg_new.shape[0] # number of grid dimensions
        for i,X in enumerate(Xg_new): # loop over grid dimensions
            assert X.shape[1] == self.grid_sub_dim[i] # number of sub-dimensions must match along each grid dimension

         # ensure that the internal state is consistent by calling parameters
        self.parameters
        if self._alpha is None: # then I need to re-fit
            self.fit()

        # get cross covariance between training and testing points
        Khat = self.kern.cov_grid(x=Xg_new, z=self.Xg, dim_noise_var=0.)

        # predict the mean at the test points
        Yhat = Khat*self._alpha
        Yhatg = Yhat.reshape([X.shape[0] for X in Xg_new], order='F') # reshape to grid shape

        # predict the variance at the test points
        if compute_var is not None:
            #raise NotImplementedError('need to do this in a way that reshuffles to fortran contiguous order, see #11')
            K = self.kern.cov_grid(x=self.Xg, dim_noise_var=self.dim_noise_var) # covariance between training data
            (Q,eig_vals) = K.svd()
            eig_vals = eig_vals.expand()

            # loop through each column of Khat and solve a linear system, exploiting kron, algebra
            n_pred = int(Khat.shape[0])
            K_post = self.kern.cov_grid(x=Xg_new, dim_noise_var=self.dim_noise_var).expand() + self.noise_var*np.eye(n_pred) # start with the covariance between testing points
            e_is = np.identity(n_pred, dtype=bool) # bunch of unit vectors
            KhatT = Khat.T
            for i in range(n_pred):
                col = KhatT * e_is[:,(i,)]
                K_post[:,(i,)] -= Khat * Q.solve_schur(eig_vals, col, shift=self.noise_var)

            # get the diagonal if thats all we want
            if compute_var == 'diag':
                logger.debug('could be more efficient if only care about diag(cov), see #5')
                K_post = K_post.diag().reshape((-1,1))
            elif compute_var == 'full':
                pass # keep the output as the full matrix
            else:
                raise ValueError('Unknown compute_var = %s' % repr(compute_var))
            return Yhatg,K_post
        else: # just return the mean
            return Yhatg


    def _adjoint_gradient(self,parameters):
        """ compute the log likelihood and the gradient wrt the hyperparameters using the adjoint method """
        # TODO: need to make more efficient that the GappyRegression method since the eigenvalue derivatives can be easily calculated analytically (see Saatci PhD, 2011)
        assert isinstance(parameters,np.ndarray)
        # get the indicies of the free parameters
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]

        # first take a forward step in each direction and get the full covariance matrix
        step = 1e-6 # finite difference step
        K_fs = np.empty(free_inds.size, dtype=object)
        noise_var_fs = np.zeros(free_inds.size) # TODO: treat the noise var as a special case so don't have to compute/factorize cov in the forward step for no reason
        for i,param_idx in enumerate(free_inds):
            p_fs = parameters.copy()
            p_fs[param_idx] += step # take a step forward
            self.parameters = p_fs # internalize these parameters
            K_fs[i] = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var) # compute the covariance matrix
            noise_var_fs[i] = self.noise_var # save the (possibly changed) noise var

        # compute the log like at the current point. the internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # compute the gradient of the log determinant by finite differentiation
        log_det_fs = np.zeros(free_inds.size)
        for i,param_idx in enumerate(free_inds):
            # compute the nystrom  approximation of the eigenvalues from the full covariance matrix
            eigs = K_fs[i].eig_vals()
            # compute the log determinant at the forward step
            log_det_fs[i] = np.sum(np.log(eigs.expand() + noise_var_fs[i])) # TODO: this is a bit wasteful, maybe do the same way as for rank-reduced where you don't use all the eigenvalues
        log_det_grad = (log_det_fs-self._log_det)/step # compute the gradient

        # compute the gradient of the y^T(K^{-1})y term, the data_term by the adjoint method
        data_term_grad = np.array([-(self._alpha.T.dot(K_fs[i]*self._alpha + noise_var_fs[i]*self._alpha - self.Y))/step
                                   for i,param_idx in enumerate(free_inds)]).squeeze()

        # compute the gradient of the log likelihood
        gradient = np.zeros(parameters.shape)
        gradient[free_inds] = -0.5*log_det_grad - 0.5*data_term_grad # set the gradient of the free indicies
        return log_like, gradient


class GPGappyRegression(GPGridRegression):
    """ similar to GridRegression but when there are gaps (missing response values) """

    def __init__(self,*args,**kwargs):
        """
        see GridRegression for inputs
        """
        # call init of the super method
        super(GPGappyRegression, self).__init__(*args,**kwargs)

        # add to dependent attributes
        self.dependent_attributes = np.unique(np.concatenate(
            (self.dependent_attributes,
             [
              '_K','_log_det','_all_eig_vals', # stuff for the marginal likelihood
              '_Q','_Tp','_big_eig_vals','_A_decomp','_S', # cov rank reduced stuff
              '_Abar_decomp','_Sbar', '_zeta','_small_eig_vals', # FG preconditioner stuff
             ])))

        # figure out where the gaps are
        self.gaps = np.isnan(self.Y).squeeze()
        self.not_gaps = np.logical_not(self.gaps)
        self.n_gaps = np.count_nonzero(self.gaps)
        self.n_not_gaps = self.num_data - self.n_gaps

        # get the interpolation matricies
        self.W = SelectionMatrix(self.not_gaps)
        self.V = SelectionMatrix(self.gaps)

        # set some defaults
        self.n_eigs = 1000 # keep the first n_eigs eigenvalues (at most) for the rank-reduced approximation
        self.preconditioner = 'rank-reduced' # options are {'rank-reduced', None, 'wilson'} (wilson only for PG)
        self.pcg_options = {'maxiter':10000, 'tol':1e-6}
        self.MLE_method = 'iterative' # options are {'rank-reduced','iterative'} rank-reduced is one shot while others computes using an iterative pcg solver.
        self.iterative_formulation = 'IG' # can be  IG - ignore gaps, FG - fill gaps, PG - penalize gaps
        self.grad_method = 'adjoint'
        self.PG_penalty = 10000 # penalty for the penalize gaps (PG) method (academic only)
        self.FG_precon_shift_frac = 0.5 # position wihin feasible range of the spectral shift for the FG rank-reduced preconditioner

        # get some counts
        self.N = self.n_not_gaps
        self.M = self.num_data
        return


    def checkgrad(self, **kwargs):
        """ chances pcg settings before running the BaseModel checkgrad routine"""
        orig_pcg_tol = self.pcg_options['tol']
        self.pcg_options['tol'] = 1e-20 # needs to be set very high for the finite-difference derivatives to be accurate
        output = super(GPGappyRegression, self).checkgrad(**kwargs) # call the super method
        self.pcg_options['tol'] = orig_pcg_tol # reset this
        return output


    def _compute_log_likelihood(self, parameters, exact_det=False):
        """
        compute log likelihood
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements are the kernel parameters
            exact_det : if True then will compute exactly (very expensive), if False then will use a Nystrom approximation

        Outputs:
            log_likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # compute the weight vector alpha
        if self.MLE_method == 'iterative': # use the exact alpha by pcg
            self.fit()
            alph = self._alpha[self.not_gaps]
        elif self.MLE_method == 'rank-reduced': # use the rank reduced approximation
            # check if I need to form the matricies required
            if self._A_decomp is None:
                self._rank_reduced_cov()
            alph = self._mv_rr_cov_inv(self.Y[self.not_gaps])
        else:
            assert False # unknown MLE_method

        # compute the log determinant of the covariance matrix
        if exact_det:
            logger.debug('using the exact log_det for log likelihood. May be expensive!')
            K = self._K.expand()[self.not_gaps,:][:,self.not_gaps]
            self._log_det = np.linalg.slogdet(K)[1]
            #print mvn.logpdf(self.Y[self.not_gaps].squeeze(),np.zeros(self.n_not_gaps),K + self.noise_var * np.eye(self.n_not_gaps))
        elif self.MLE_method == 'iterative': # use the exact alpha by pcg
            # compute the nystrom  approximation of the eigenvalues from the full covariance matrix
            if self._all_eig_vals is not None: # then they are already computed and expanded
                nystrom_eigs = np.partition(a=self._all_eig_vals, kth=-self.N)[-self.N:] # get the N largest eigenvalues
            else: # then compute the largest N eigenvalues
                nystrom_eigs = self._K.eig_vals().find_extremum_eigs(n_eigs=self.N, mode='largest', log_expand=False, sort=False, compute_global_loc=False)[1]
            nystrom_eigs *= np.float64(self.N)/self.M # scale the eigenvalues appropriately
            # compute the log determinant
            self._log_det = np.sum(np.log(nystrom_eigs + self.noise_var))
        elif self.MLE_method == 'rank-reduced': # use the rank reduced approximation
            # get the approximate eigenvalues of the covariance matrix (which were already computed)
            nystrom_eigs = self._big_eig_vals*(np.float64(self.N)/self.M)
            assert nystrom_eigs.size <= self.N
            self._log_det = np.sum(np.log(nystrom_eigs + self.noise_var)) + \
                    (self.n_not_gaps-nystrom_eigs.size)*np.log(self.noise_var)# these are the zero eigenvalues

        # compute the log likelihood
        log_like = \
                   -0.5*self._log_det \
                   -0.5*np.dot(self.Y[self.not_gaps].T, alph) \
                   -0.5*self.n_not_gaps*np.log(np.pi*2)
        return log_like.squeeze()


    def _rank_reduced_cov(self):
        """
        constructs all matricies required to compute the rank reduced covariance matrix
        """
        # get the covariance matrix
        self._K = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var)

        # compute svd
        (self._Q,T) = self._K.schur()
        all_eig_vals = T.diag()

        # get the biggest eigenvalues and eigenvectors
        n_eigs = min(self.n_eigs, all_eig_vals.shape[0]) # can't use more eigenvalues then all of them
        eig_pos, self._big_eig_vals, big_eig_idx = all_eig_vals.find_extremum_eigs(n_eigs=n_eigs,mode='largest', compute_global_loc=True, log_expand=False, sort=False)

        # compute A = S Q^T W^T W Q S^T
        A = np.zeros((n_eigs,n_eigs))
        for i in range(n_eigs):
            # compute Q S^T column by column. This is basically just selecting the current active eigenvalue
            q = self._Q.get_col(eig_pos[i]).expand()
            # compute q = W^T W q which simply zeros the values in q corresponding to the gaps
            q[self.gaps] = 0
            # compute q = Q^T q
            q = self._Q.T.kronvec_prod(q)
            # compute a = S q which is basically just selecting specific elements of q in a specific order
            A[:,i] = q[big_eig_idx].squeeze()

        # compute A = Tp A + sig^2 I
        self._Tp = sparse.diags(self._big_eig_vals)
        I_sig2 = self.noise_var*sparse.identity(n_eigs)
        A = self._Tp * A + I_sig2

        # initialize the explicit selection matrix S which is useful for performing the mvprod with the rank reduced matrix
        self._S = SelectionMatrix(indicies=(big_eig_idx,self.M))

        # perform LU decomposition of A
        self._A_decomp = lu_factor(a=A, overwrite_a=True, check_finite=False)


    def _fg_preconditioner(self):
        """
        constructs all matricies required for the rank-reduced preconditioner for the fill-gaps formulation
        """
        # get the covariance matrix
        self._K = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var)

        # compute svd
        (self._Q,T) = self._K.schur()
        all_eig_vals = T.diag()
        self._all_eig_vals = all_eig_vals.expand() # save the expanded version since will need it for FG anyway

        # get the smallest eigenvalues and eigenvectors
        n_eigs = np.int64(min(self.n_eigs, all_eig_vals.shape[0])) # can't use more eigenvalues then all of them
        eig_pos, self._small_eig_vals, small_eig_idx = all_eig_vals.find_extremum_eigs(n_eigs=n_eigs,mode='smallest', compute_global_loc=True, log_expand=False, sort=False)

        # determine what the spectral shift should be
        self._zeta = self.FG_precon_shift_frac / (self._small_eig_vals.max() + self.noise_var)

        # compute Abar = Tp_bar Sbar Q^T V^T V Q Sbar^T + ... additional term later
        Abar = np.zeros((n_eigs,n_eigs))
        for i in range(n_eigs):
            # compute Q Sbar^T column by column. This is basically just selecting the current active eigenvalue
            q = self._Q.get_col(eig_pos[i]).expand()
            # compute q = V^T V q which simply zeros the values in q corresponding the where there is not gaps
            q[self.not_gaps] = 0
            # compute q = Q^T q
            q = self._Q.T.kronvec_prod(q)
            # compute q = Sbar q which is basically just selecting specific elements of q in a specific order
            q = q[small_eig_idx].squeeze()
            # now compute a = Tp_bar q
            Abar[:,i] = q / (self._small_eig_vals + self.noise_var) - self._zeta * q

        # compute Abar = Abar + zeta I
        Abar += self._zeta*sparse.identity(n_eigs)

        # initialize the explicit selection matrix Sbar which is useful for performing the mvprod with the rank-reduced matrix
        self._Sbar = SelectionMatrix(indicies=(small_eig_idx,self.M))

        # perform LU decomposition of Abar
        self._Abar_decomp = lu_factor(a=Abar, overwrite_a=True, check_finite=False)


    def _mv_rr_cov_inv(self,x):
        """
        define a function to perform a matrix vector product with the inverse of the rank-reduced covariance matrix
        This computes:
            y = (I x - W Q S.T A^{-1}Tp S Q.T W^T x)/sig^2
        """
        assert x.size == self.n_not_gaps
        assert self._A_decomp is not None # ensure this has been precomputed
        x = x.reshape((-1,1)) # ensure it is a 2d vector
        y = (x - self.W.mul(self._Q*(self._S.mul_T(lu_solve(self._A_decomp, trans=0, overwrite_b=False, check_finite=False,
                b=self._Tp*(self._S.mul(self._Q.T*(self.W.mul_T(x))))))))) / self.noise_var
        return y


    def _mv_rr_fg_precon(self,x):
        """
        performs matrix-vector product with the Fill-gaps rank-reduced preconditioner
        This computes:
            y = (I x - V Q Sbar.T Abar^{-1} Tp_bar Sbar Q.T V^T x) / zeta
        """
        assert x.size == self.n_gaps
        assert self._Abar_decomp is not None # ensure this has been precomputed
        x = x.reshape((self.n_gaps,1)) # ensure it is a 2d vector
        # first compute y = Sbar Q.T V^T x
        y = self._Sbar.mul(self._Q.T*(self.V.mul_T(x)))
        # then compute y = (I x - V Q Sbar.T Abar^{-1} Tp_bar y) / zeta
        y = (x - self.V.mul(self._Q*(self._Sbar.mul_T(lu_solve(self._Abar_decomp, trans=0, overwrite_b=True, check_finite=False,
                b=y/(self._small_eig_vals.reshape((-1,1)) + self.noise_var)-self._zeta*y))))) / self._zeta
        return y


    def fit(self, use_counter=False):
        """
        finds the weight vector alpha using a linear pcg solver

        use_counter : prints additional debug info and prints value at each iteration
        """
        if use_counter: logger.debug('Fitting; determining weight vector.')
        self.parameters # ensure that the internal state is consistent!

        # check if alpha is already computed
        if self._alpha is not None: # then already computed
            if use_counter: logger.debug('... alpha has already been computed.')
            return

        # setup the preconditioner
        if self.preconditioner == 'rank-reduced':
            if self.iterative_formulation not in ['IG','FG'] :
                raise ValueError('the rank-reduced preconditioner is for the IG or PG formulations only.')
            elif self.iterative_formulation == 'IG':
                # check if I need to form the matricies required for the preconditioner
                if self._A_decomp is None:
                    self._rank_reduced_cov()
                precon = LinearOperator(shape=(self.n_not_gaps,)*2, matvec=self._mv_rr_cov_inv)
            elif self.iterative_formulation == 'FG':
                # check if I need to form the matricies required for the preconditioner
                if self._Abar_decomp is None:
                    self._fg_preconditioner()
                precon = LinearOperator(shape=(self.n_gaps,)*2, matvec=self._mv_rr_fg_precon)
        elif self.preconditioner is None:
            precon = None
        elif self.preconditioner == 'wilson': # wilson's precon for the PG formulation
            if self.iterative_formulation != 'PG':
                raise ValueError('the wilson preconditioner is for the PG formulation only.')
            pass # will set this up below
        else:
            raise ValueError('unknown precon %s'% repr(self.preconditioner))
        if use_counter: logger.debug('using preconditioner %s' % repr(self.preconditioner))

        # get covariance matrix if ness
        if self._K is None:
            self._K = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var)

        # fit the model
        if use_counter:
            counter = solver_counter(disp=True) # use this counter if you want the number of iterations displayed in debug logger (SLOW!)
        else:
            counter = None
        if self.iterative_formulation == 'IG': # ignore gaps
            # compute by solving a linear system of equations
            self._alpha = np.zeros((self.M,1)) # I need alpha to be zero wherever there is a gap
            self._alpha[self.not_gaps,0],info = pcg( A=LinearOperator(shape=(self.n_not_gaps,)*2, matvec=self._mv_IG),
                                      b=self.Y[self.not_gaps],
                                      M=precon,
                                      callback=counter,
                                      **self.pcg_options)
        elif self.iterative_formulation == 'FG': # fill gaps
            # factorize the full covariance matrix
            if self._Q is None: # check if already computed (if using a rr precon then will be)
                (self._Q,T) = self._K.schur()
                self._all_eig_vals = T.diag().expand()
            # get the response vector: -V Q (T + sig^2 I)^{-1} Q^T W^T y[not_gaps]
            b = -(self.V.mul(self._Q.solve_schur(t=self._all_eig_vals,shift=self.noise_var,x=self.W.mul_T(self.Y[self.not_gaps]))))
            # solve for the missing values on the gaps by pcg
            missing_vals,info = pcg( A=LinearOperator(shape=(self.n_gaps,)*2, matvec=self._mv_FG),
                                      b=b,
                                      M=precon,
                                      callback=counter,
                                      **self.pcg_options)
            Y_filled = self.Y.copy()
            Y_filled[self.gaps,0] = missing_vals # replace the missing values
            # compute the response vector alpha by solving the full equations
            self._alpha = self._Q.solve_schur(t=self._all_eig_vals,shift=self.noise_var,x=Y_filled)
        elif self.iterative_formulation == 'PG':
            diag = np.ones((self.M,1))*self.noise_var # diag is the the diagonal of the matrix formed by (sig^2 I + R)
            diag[self.gaps] = self.PG_penalty # apply the penalty to the gaps
            if self.preconditioner == 'wilson': # then set up the preconditioner
                sqrt_diag = np.sqrt(diag)
                precon = LinearOperator(shape=(self.M,)*2,
                                        matvec=lambda x: x.reshape((-1,1))/sqrt_diag)
            else:
                precon = None; logger.debug('not using any preconditioner for PG')
            b = self.Y.copy();      b[self.gaps] = 0 # set the values on the gaps to an arbitrary value
            self._alpha,info = pcg(A=LinearOperator(shape=(self.M,)*2,
                                        matvec=lambda x: self._K*(x.reshape((-1,1))) + diag*x.reshape((-1,1))),
                                      b=b,
                                      M=precon,
                                      callback=counter,
                                      **self.pcg_options)
            self._alpha = self._alpha.reshape((-1,1))
        else:
            assert False # unknown formulation

        if info == 0 and counter is not None:
            logger.debug('pcg successfully converged in %d iterations (max_iter=%d)' % (counter.niter,self.pcg_options['maxiter']))
        elif info > 0:
            logger.critical('pcg convergence to tolerance not achieved. Number of iterations: %d' % info)
        elif info < 0:
            logger.critical('pcg illegal input or breakdown')

        # save the precon
        self.precon = precon


    def _mv_IG(self,x):
        """
        matrix vector product with the covariance matrix
        This computes:
            y = (W K W^T + sig^2 I) x
        """
        x = x.reshape((-1,1))
        #y = self.W * (self._K.kronvec_prod(self.W.T * x)) + self.noise_var * x
        y = self.W.mul_T(x)
        y = (self._K.kronvec_prod(y))
        y = self.W.mul(y)
        y+= self.noise_var * x
        return y


    def _mv_FG(self,x):
        """
        compute the matrix-vector product with the partition of the inverse of the full covariance matrix for
        the FG (fill-gaps) formulation. This computes:
            y = V (K + sig^2 I)^{-1} V^T x
        """
        y = self.V.mul(self._Q.solve_schur(t=self._all_eig_vals,shift=self.noise_var,x=self.V.mul_T(x.reshape((-1,1)))))
        return y


    def predict_grid(self,Xg_new,compute_var=None):
        if compute_var is not None:
            raise NotImplementedError('see self.predict_cov')
        return super(GPGappyRegression, self).predict_grid(Xg_new,compute_var)


    def predict_cov(self, X_new, exact=True):
        """
        predict the posterior covariance at a single point
        if not exact then will use a rank reduced approximation
        X_new is a [n_test,d] matrix
        """
        assert isinstance(X_new, np.ndarray)
        assert X_new.ndim == 2
        assert X_new.shape[1] == self.input_dim, "wrong shape: %s" % X_new.shape
        n_test = X_new.shape[0]
        self.parameters # ensure that the internal state is consistent!

        # setup the rank-reduced approximation if ness 
        if not exact or self.preconditioner == 'rank-reduced':
            if not hasattr(self,'precon') or "IG" not in self.iterative_formulation: # then need to initialize the preconditioner, else just use what we had before
                logger.info('forming preconditioner to predict posterior variance')
                # check if I need to form the matricies required for the preconditioner
                if self._A_decomp is None:
                    self._rank_reduced_cov()
                self.precon = LinearOperator(shape=(self.n_not_gaps,)*2, matvec=self._mv_rr_cov_inv)
        elif exact: # then assign the preconditioner to the defualt
            assert self.preconditioner is None, "unknown preconditioner: %s" % repr(self.preconditioner)
            self.precon = None

        # get covariance matrix if ness
        if self._K is None:
            self._K = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var)

        # compute the prior covariance between test points
        Kp = self.kern.cov(X_new) + self.noise_var*np.identity(n_test)

        # loop through each test point and solve the linear system
        G = np.zeros((self.n_not_gaps,n_test)) # cross covariance matrix
        KiG = np.zeros((self.n_not_gaps,n_test)) # (K + sig^2 I)^{-1} G
        for i_pnt,x in enumerate(X_new):
            x = x.reshape((1,-1))

            # convert the test point to grid form
            xg = np.empty(self.grid_dim, dtype=object)
            i_cur = 0
            for i,n_dim in enumerate(self.grid_sub_dim):
                xg[i] = x[0,None,i_cur:(i_cur+n_dim)]
                i_cur += n_dim

            # compute the cross covariance vector
            G[:,i_pnt] =  self.kern.cov_grid(self.Xg, xg, dim_noise_var=0.).expand()[self.not_gaps,0]

            # compute the posterior covariance; first solve the linear system
            if exact: # then solve system using linear CG
                KiG[:,i_pnt],info = pcg( A=LinearOperator(shape=(self.n_not_gaps,)*2, matvec=self._mv_IG),
                                          b=G[:,i_pnt],
                                          M=self.precon,
                                          callback=None,
                                          **self.pcg_options)
                if info > 0:
                    logger.critical('pcg convergence to tolerance not achieved. Number of iterations: %d' % info)
                elif info < 0:
                    logger.critical('pcg illegal input or breakdown')
            else: # use the rank reduced approximation
                KiG[:,i_pnt,None] = self._mv_rr_cov_inv(G[:,i_pnt,None])

        # ... then complete the computation
        Kp = Kp - G.T.dot(KiG)
        return Kp


    def _adjoint_gradient(self, parameters):
        """ compute the log likelihood and the gradient wrt the hyperparameters using the adjoint method """
        assert isinstance(parameters,np.ndarray)
        if self.MLE_method != 'iterative':
            raise NotImplementedError('have only done this for the iterative method. In the future would like to possibly do for rank-reduced')
        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]

        # first take a forward step in each direction and get the full covariance matrix
        step = 1e-6 # finite difference step
        K_fs = np.empty(free_inds.size, dtype=object)
        noise_var_fs = np.zeros(free_inds.size) # TODO: treat the noise var as a special case so don't have to compute/factorize cov in the forward step for no reason
        for i,param_idx in enumerate(free_inds):
            p_fs = parameters.copy()
            p_fs[param_idx] += step # take a step forward
            self.parameters = p_fs # internalize these parameters
            K_fs[i] = self.kern.cov_grid(self.Xg, dim_noise_var=self.dim_noise_var) # compute the covariance matrix
            noise_var_fs[i] = self.noise_var # save the (possibly changed) noise var

        # compute the log like at the current point. the internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # compute the gradient of the log determinant by finite differentiation
        assert self.MLE_method != 'rank-reduced', "TODO: need to change the log-det for rank-reduced to use just p eigvals (+ the other zeros)"
        log_det_fs = np.zeros(free_inds.size)
        for i,param_idx in enumerate(free_inds):
            # compute the nystrom  approximation of the eigenvalues from the full covariance matrix
            nystrom_eigs = K_fs[i].eig_vals().find_extremum_eigs(n_eigs=self.N, mode='largest', compute_global_loc=False, log_expand=False, sort=False)[1]
            nystrom_eigs *= np.float64(self.N)/self.M # scale the eigenvalues appropriately
            # compute the log determinant at the forward step
            log_det_fs[i] = np.sum(np.log(nystrom_eigs + noise_var_fs[i]))
        log_det_grad = (log_det_fs-self._log_det)/step # compute the gradient

        # compute the gradient of the y^T(K^{-1})y term, the data_term by the adjoint method
        # ... note that since alpha is zero on the gaps I don't need to care about their effect. They don't change the gradient
        y_pred = self._K*self._alpha + self.noise_var*self._alpha # the prediction of the model at the training points (plus on the gaps)
        data_term_grad = np.array([-(self._alpha.T.dot(K_fs[i]*self._alpha + noise_var_fs[i]*self._alpha - y_pred))/step 
                                   for i,param_idx in enumerate(free_inds)]).squeeze()

        # compute the gradient of the log likelihood
        gradient = np.zeros(parameters.shape)
        gradient[free_inds] = -0.5*log_det_grad - 0.5*data_term_grad # set the gradient of the free indicies
        return log_like, gradient


