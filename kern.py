from .tensors import KronMatrix
import numpy as np
from numpy import pi
from itertools import product
import logging
import GPy.kern
from copy import deepcopy
logger = logging.getLogger(__name__)
from pdb import set_trace


class BaseKernel(object):
    """ base class for all kernel functions """

    def __init__(self, n_dims, active_dims, name):
        self.n_dims = n_dims
        if active_dims is None: # then all dims are active
            active_dims = np.arange(self.n_dims)
        else: # active_dims has been specified
            active_dims = np.ravel(active_dims) # ensure 1d array
            assert 'int' in active_dims.dtype.type.__name__ # ensure it is an int array
            assert active_dims.min() >= 0 # less than zero is not a valid index
            assert active_dims.max() <  self.n_dims # max it can be is n_dims-1
        self.active_dims = active_dims
        if name is None: # then set a default name
            name = self.__class__.__name__
        self.name = name

        # initialize a few things that need to be implemented for new kernels
        self.parameter_list = None # list of parameter attribute names as strings
        self.constraint_map = None # dict with elements in parameter_list as keys


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        x,z = self._process_cov_inputs(x,z) # process inputs
        raise NotImplementedError('Not implemented')

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        # check if implemented
        if self.parameter_list is None:
            raise NotImplementedError('Need to specify kern.parameter_list')

        # get the parameters
        parameters = [np.ravel(getattr(self, name)) for name in self.parameter_list]

        # now concatenate into an array
        if len(parameters) > 0:
            parameters = np.concatenate(parameters, axis=0)
        else:
            parameters = np.array([])
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d
        i0 = 0 # counter of current position in value

        # set the parameters
        for name in self.parameter_list:
            old = getattr(self, name) # old value
            setattr(self, name, value[i0:i0+np.size(old)].reshape(np.shape(old))) # ensure same shape as old
            i0 += np.size(old) # increment counter

    @property
    def constraints(self):
        """ returns the constraints for all parameters """
        # check if implemented
        if self.constraint_map is None:
            raise NotImplementedError('Need to specify kern.constraint_map')

        # get the parameters
        constraints = [np.ravel(self.constraint_map[name]) for name in self.parameter_list]

        # now concatenate into an array
        if len(constraints) > 0:
            constraints = np.concatenate(constraints, axis=0)
        else:
            constraints = np.array([])
        return constraints


    def is_stationary(self):
        """ check if stationary """
        if isinstance(self, GPyKernel):
            return isinstance(self.kern, GPy.kern.src.stationary.Stationary)
        else:
            return isinstance(self, Stationary)


    def _process_cov_inputs(self,x,z):
        """
        function for processing inputs to the cov function

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d). If none then will assume z=x

        Outputs:
            x,z
        """
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims
        if z is None:
            z = x
        else:
            assert z.ndim == 2
            assert z.shape[1] == self.n_dims, "should be %d dims, not %d" % (self.n_dims,z.shape[1])
        return x,z


    def __str__(self):
        """
        this is what is used when being printed
        """
        from tabulate import tabulate
        # print the parent
        s = '\n'
        s += "%s kernel\n" % self.name
        if isinstance(self, GPyKernel): # do this in a custom way
            s += str(tabulate([[param._name, param.values, constraint]
                               for (param, constraint) in zip(self.kern.flattened_parameters, self.constraint_list)],
                              headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl'))
        elif isinstance(self, DirectCovariance): # then print in a custom way
            formatter = {'float_kind':lambda x: '%5.2f'%x, 'str_kind':lambda x: "{:<5}".format(x)}
            s += "L:\n"
            #s += np.array2string(a=self.L, max_line_width=np.inf, formatter=formatter)
            s += str(self.L) # for some reason the above isn't working
            s += '\n'
            #s += np.array2string(a=self.constraint_map['L'], max_line_width=np.inf, formatter=formatter)
            s += str(self.constraint_map['L']) # for some reason the above isn't working
            s += '\nK:\n'
            #s += np.array2string(a=self.L.dot(self.L.T), max_line_width=np.inf, formatter=formatter)
            s += str(self.L.dot(self.L.T)) # for some reason the above isn't working
        else: # tabulate the reuslts
            s += str(tabulate([[name, getattr(self, name), self.constraint_map[name]] for name in self.parameter_list],
                              headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl'))
        s += '\n'
        return s


    def copy(self):
        """ return a deepcopy """
        # first create a deepcopy
        self_copy = deepcopy(self)
        return self_copy


class GPyKernel(BaseKernel):
    """ grab some kernels from the GPy library """

    def __init__(self, n_dims, kernel=None, name=None, **kwargs):
        """
        Use a kernel from the GPy library

        Inputs:
            n_dims : int
                number of input dimensions
            kernel : str OR GPy.kern.Kern
                name of the kernel in gpy OR GPy kernel object. If the latter then nothing else afterwards should be specified
                except name can be
        """
        if isinstance(kernel, str):
            if name is None:
                name = "GPy - " + kernel
            super(GPyKernel, self).__init__(n_dims=n_dims, active_dims=None, name=name) # Note that active_dims will be dealt with at the GPy level
            logger.debug('Initializing %s kernel.' % self.name)
            self.kern = eval("GPy.kern." + kernel)(input_dim=n_dims,**kwargs) # get the kernel
        elif isinstance(kernel, GPy.kern.Kern): # check if its a GPy object
            if name is None:
                name = "GPy - " + repr(kernel)
            super(GPyKernel, self).__init__(n_dims=n_dims, active_dims=None, name=name) # Note that active_dims will be dealt with at the GPy level
            logger.debug('Using specified %s GPy kernel.' % self.name)
            self.kern = kernel
        else:
            raise TypeError("must specify kernel as string or a GPy kernel object")

        # Constrain parameters  TODO: currently assuming all parameters are constrained positive, I should be able to take this directly from the flattened_parameters
        self.constraint_list = [['+ve',]*np.size(param.values) for param in self.kern.flattened_parameters]


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        K = self.kern.K(x,z)
        return K

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        # first get parameters as a list
        parameters = [np.ravel(param.values) for param in self.kern.flattened_parameters]

        # now concatenate into an array
        if len(parameters) > 0:
            parameters = np.concatenate(parameters, axis=0)
        else:
            parameters = np.array([])
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d
        i0 = 0 # counter of current position in value

        # set the parameters
        for ip in range(np.size(self.kern.flattened_parameters)):
            old = self.kern.flattened_parameters[ip]
            try:
                self.kern.flattened_parameters[ip][:] = value[i0:i0+np.size(old)].reshape(np.shape(old)) # ensure same shape as old
            except:
                raise
            i0 += np.size(old) # increment counter


    @property
    def constraints(self):
        """ get constraints. over ride of inherited property """
        constraints = [np.ravel(constraint) for constraint in self.constraint_list]

        # now concatenate into an array
        if len(constraints) > 0:
            constraints = np.concatenate(constraints, axis=0)
        else:
            constraints = np.array([])
        return constraints


    def fix_variance(self):
        """ apply fixed constraint to the variance """
        # look for the index of each occurance of variance
        i_var = np.where(['variance' in param._name.lower() for param in self.kern.flattened_parameters])[0]

        # check if none or multiple found
        if np.size(i_var) == 0:
            raise RuntimeError("No variance parameter found")
        elif np.size(i_var) >  1 or np.size(self.constraint_list[i_var[0]]) > 1:
            # ... this should be valid even when the kernel is eg. a sum of other kernels
            logger.info("Multiple variance parameters found in the GPy kernel, will only fix the first")

        # constrain it
        self.constraint_list[i_var[0]][0] = 'fixed'


class Stationary(BaseKernel):
    """ base class for stationary kernels """

    def distances_squared(self, x, z=None, lengthscale=None):
        """
        Evaluate the distance between points squared.

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional)

        Outputs:
            k : matrix of distances of shape shape (N, M)
        """
        x, z = self._process_cov_inputs(x, z) # process inputs

        # reshape the matricies correctly for broadcasting
        N = x.shape[0]
        M = z.shape[0]
        d = self.active_dims.size # the number of active dimensions
        x = np.asarray(x)[:,self.active_dims].reshape((N,1,d))
        z = np.asarray(z)[:,self.active_dims].reshape((1,M,d))

        # Code added to use different lengthscales for each dimension
        if lengthscale is None:
            lengthscale = np.ones(d,dtype='d')
        elif isinstance(lengthscale,float):
            lengthscale = lengthscale*np.ones(d,dtype='d')
        else:
            lengthscale = np.asarray(lengthscale).flatten()
            assert len(lengthscale) == d

        # now compute the distances
        return np.sum(np.power((x-z)/lengthscale.reshape((1,1,d)),2),
                      axis=2, keepdims=False)


    def distances(self, x, z=None, lengthscale=None):
        """
        Evaluate the distance between points along each dimension

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional)

        Outputs:
            k : matrix of distances of shape (N, M, d)
        """
        x, z = self._process_cov_inputs(x, z) # process inputs

        # reshape the matricies correctly for broadcasting
        N = x.shape[0]
        M = z.shape[0]
        d = self.active_dims.size # the number of active dimensions
        x = np.asarray(x)[:,self.active_dims].reshape((N,1,d))
        z = np.asarray(z)[:,self.active_dims].reshape((1,M,d))

        # Code added to use different lengthscales for each dimension
        if lengthscale is None:
            lengthscale = np.ones(d,dtype='d')
        elif isinstance(lengthscale,float):
            lengthscale = lengthscale*np.ones(d,dtype='d')
        else:
            lengthscale = np.asarray(lengthscale).flatten()
            assert len(lengthscale) == d

        # now compute the distances
        return (x-z)/lengthscale.reshape((1,1,d))


class RBF(Stationary):
    """squared exponential kernel with the same shape parameter in each dimension"""

    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs: (very much the same as in GPy.kern.RBF)
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(RBF, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        assert np.size(variance) == 1
        assert np.size(lengthscale) == 1
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None,lengthscale=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x
            lengthscale : a vector of length scales for each dimension

        Outputs:
            k : matrix of shape (N, M)
        """
        if self.lengthscale < 1e-6: # then make resiliant to division by zero
            K = self.variance * (self.distances_squared(x=x,z=z)==0) # the kernel becomes the delta funciton (white noise)
            logger.debug('protected RBF against zero-division since lengthscale too small (%s).' % repr(self.lengthscale))
        else: # then compute the nominal way
            if lengthscale is None:
                K = self.variance * np.exp( -0.5 * self.distances_squared(x=x,z=z) / self.lengthscale**2 )
            else:
                lengthscale = np.asarray(lengthscale).flatten()
                assert len(lengthscale) == self.active_dims.size
                K = self.variance * np.exp( -0.5 * self.distances_squared(x=x,z=z,lengthscale=lengthscale) )
        return K


class Exponential(Stationary):
    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(Exponential, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r = np.sqrt(self.distances_squared(x=x,z=z)) / self.lengthscale
        K = self.variance * np.exp( -r )
        return K


class Matern32(Stationary):
    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(Matern32, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r = np.sqrt(self.distances_squared(x=x,z=z)) / self.lengthscale
        K = self.variance * (1.+np.sqrt(3.)*r) * np.exp(-np.sqrt(3.)*r)
        return K


class Matern52(Stationary):
    def __init__(self, n_dims, variance=1., lengthscale=1., active_dims=None, name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : by default all dims are active but this can instead be a subset specified
                as a list or array of ints
        """
        super(Matern52, self).__init__(n_dims=n_dims, active_dims=active_dims, name=name)
        logger.debug('Initializing %s kernel.' % self.name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def cov(self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r2 = self.distances_squared(x=x,z=z) / self.lengthscale**2
        r = np.sqrt(r2)
        K = self.variance * (1.+np.sqrt(5.)*r+(5./3)*r2) * np.exp(-np.sqrt(5.)*r)
        return K


class GridKernel(object):
    """ simple wrapper for a kernel for GridRegression which is a product of 1d kernels """
    def __init__(self, kern_list, radial_kernel=False):
        """
        Kernel for gridded inducing point methods and structured problems

        Inputs:
            kern_list : list or 1d array of kernels
            radial_kernel : bool
                if true then will use the same kernel along each dimension. Will just grab the kernel from the first dimension to use for all.
        """
        # initialize the kernel list
        self.kern_list = kern_list

        # add the dimension of the grid
        self.grid_dim = len(kern_list)

        # check if radial kernel
        assert isinstance(radial_kernel, bool)
        self.radial_kernel = radial_kernel
        if self.radial_kernel:
            for kern in self.kern_list:
                assert kern.n_dims == self.kern_list[0].n_dims, "number of grid dims must be equal for all slices"
            self.kern_list = [self.kern_list[0],]*np.size(kern_list) # repeat the first kernel along all dimensions
        else:
            # set the variance as fixed for all but the first kernel. 
            # ... this should be valid even when the kernel is eg. a sum of other kernels
            for i in range(1,self.grid_dim):
                if hasattr(self.kern_list[i], 'fix_variance'):
                    self.kern_list[i].fix_variance()
                elif np.size(self.kern_list[i].constraint_map['variance']) > 1:
                    _logger.info("Multiple variance parameters found in the kernel, will only fix the first")
                    self.kern_list[i].constraint_map['variance'][0] = 'fixed'
                else:
                    self.kern_list[i].constraint_map['variance'] = 'fixed'

        # the the total number of dims
        self.n_dims = np.sum([kern.n_dims for kern in self.kern_list])
        return


    def cov_grid(self, x, z=None, dim_noise_var=None):
        """
        generates a matrix which creates a covariance matrix mapping between x1 and x2.
        Inputs:
          x : numpy.ndarray of shape (self.grid_dim,)
          z : (optional) numpy.ndarray of shape (self.grid_dim,) if None will assume x2=x1
              for both x1 and x2:
              the ith element in the array must be a matrix of size [n_mesh_i,n_dims_i]
              where n_dims_i is the number of dimensions in the ith kronecker pdt
              matrix and n_mesh_i is the number of points along the ith dimension
              of the grid.
              Note that for spatial temporal datasets, n_dims_i is probably 1
              but for other problems this might be of much higher dimensions.
          dim_noise_var : float (optional)
              diagonal term to use to shift the diagonal of each dimension to improve conditioning

        Outputs:
          K : gp_grid.tensors.KronMatrix of size determined by x and z (prod(n_mesh1(:)), prod(n_mesh2(:))
              covariance matrix
        """
        assert dim_noise_var is not None, "dim_noise_var must be specified"

        # check inputs (minimal here, rest will be taken care of by calls to kern.cov)
        assert len(x) == self.grid_dim # ensure the first dimension is the same as the grid dim
        if z is None:
            cross_cov = False
            z = [None,] * self.grid_dim # array of None
        else:
            cross_cov = True
            assert len(z) == self.grid_dim # ensure the first dimension is the same as the grid dim

        # get the 1d covariance matricies
        K = []
        for i,kern in enumerate(self.kern_list): # loop through and generate the covariance matricies
            K.append(kern.cov(x=x[i],z=z[i]))

        # now create a KronMatrix instance
        K = KronMatrix(K[::-1], sym=(z[0] is None)) # reverse the order and set as symmetric only if the two lists are identical

        # shift the diagonal of the sub-matricies if required
         # TODO: really this shouldn't just be where the diagonal is for cov matricies but anywhere the covariance is evaluated btwn two identical points
        if dim_noise_var != 0.:
            assert not cross_cov, "not implemented for cross covariance matricies yet"
            K = K.sub_shift(shift=dim_noise_var)
        return K


    def cov(self,x,z=None, dim_noise_var=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        assert dim_noise_var is None, "currenly no way to add dim_noise_var to this"

        # loop through each dimension, compute the 1(ish)-dimensional covariance and perform hadamard product
        i_cur = 0
        zi = None # set default value
        for i,kern in enumerate(self.kern_list):
            xi = x[:,i_cur:(i_cur+kern.n_dims)] # just grab a subset of the dimensions
            if z is not None:
                zi = z[:,i_cur:(i_cur+kern.n_dims)]
            i_cur += kern.n_dims

            # compute the covaraince of the subset of dimensions and multipy with the other dimensions
            if i == 0:
                K = kern.cov(x=xi,z=zi)
            else: # perform hadamard product
                K = np.multiply(K, kern.cov(x=xi,z=zi))
        return K


    def cov_kr(self,x,z, dim_noise_var=None, form_kr=True):
        """
        Evaluate covariance kernel at points to form a covariance matrix in row partitioned Khatri-Rao form

        Inputs:
            x : array of shape (N, d)
            z : numpy.ndarray of shape (d,)
              the ith element in the array must be a matrix of size [n_mesh_i,1]
              where n_mesh_i is the number of points along the ith dimension
              of the grid.
            form_kr : if True will form the KhatriRao matrix, else will just return a list of arrays

        Outputs:
            k : row partitioned Khatri-Rao matrix of shape (N, prod(n_mesh))
        """
        assert dim_noise_var is None, "currenly no way to add dim_noise_var to this"
        (N,d) = x.shape
        assert self.grid_dim == d, "currently this only works for 1-dimensional grids"

        # loop through each dimension and compute the 1-dimensional covariance matricies
        # and compute the covaraince of the subset of dimensions
        Kxz = [kern.cov(x=x[:,(i,)],z=z[i]) for i,kern in enumerate(self.kern_list)]

        # flip the order
        Kxz = Kxz[::-1]

        # convert to a Khatri-Rao Matrix
        if form_kr:
            Kxz = KhatriRaoMatrix(A=Kxz, partition=0) # row partitioned
        return Kxz

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        if self.radial_kernel:
            parameters = np.ravel(self.kern_list[0].parameters)
        else:
            parameters = np.concatenate([np.ravel(kern.parameters) for kern in self.kern_list], axis=0)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d

        # set the parameters
        if self.radial_kernel:
            self.kern_list[0].parameters = value
            self.kern_list = [self.kern_list[0],]*np.size(self.kern_list) # repeat the first kernel along all dimensions
        else:
            i0 = 0 # counter of current position in value
            for kern in self.kern_list:
                # get the old parameters to check the size
                old = kern.parameters
                # set the parameters
                kern.parameters = value[i0:i0+np.size(old)].reshape(np.shape(old))
                i0 += np.size(old) # increment counter

    @property
    def constraints(self):
        """
        returns the kernel parameters' constraints as a 1d array
        """
        if self.radial_kernel:
            constraints = np.ravel(self.kern_list[0].constraints)
        else:
            constraints = np.concatenate([np.ravel(kern.constraints) for kern in self.kern_list], axis=0)
        return constraints

    @property
    def diag_val(self):
        """ return diagonal value of covariance matrix. Note that it's assumed the kernel is stationary """
        return self.cov(np.zeros((1,self.n_dims))).squeeze()


    def __str__(self):
        """ prints the kernel """
        s = '\nGridKernel'
        if self.radial_kernel:
            s += " Radial (same kern along all dimensions)\n"
            s += str(self.kern_list[0]) + '\n'
        else:
            for i,child in enumerate(self.kern_list):
                s += '\nGrid Dimension %d' % i
                s += str(child) + '\n'
        return s


class DirectCovariance(BaseKernel):
    """ directly specifying a covariance between discrete points """

    def __init__(self, L=None, K=None, X=None, name=None, fix_cross_cov=False, observation_weights=None):
        """
        Initialize kernel which creates an empirical covariance matrix based on N data observations.
        It is assumed that the size of the resultant covariance matrix is m x m

        Inputs:
            L : (m x m) Cholesky factorized empirical covariance (optional)
                if not specified then will check if K specified, else will compute from observations, X
            K : (m x m) empirical covariance (optional)
                if not specified then will compute from observations, X
            X : (N x m) observations
                data observations from which the MLE covariance will be computed
            fix_cross_cov : (optional) will fix all the cross covariance terms. This
                is useful if you want the GPs to be independent (i.e. K=np.identity(m),fix_cross_cov=True)
            observation_weights : (N,) array of observation weights (optional)

        Notes:
            * from experience, it seems initializing based on X isn't typically very helpful and starting
              uncorrelated (i.e. K = np.identity(m)) gives better results
        """
        super(DirectCovariance, self).__init__(n_dims=1, active_dims=None, name=name)
        if L is not None: # then use this as the decomposition of K
            assert np.all(np.ravel(np.triu(L,k=1)) == 0), "L must be lower triangular"
            assert np.all(np.diag(L)) > 0, "diag(L) terms must be positive"
            self.L = L
        elif K is not None: # then just store the covariance
            self.L = np.linalg.cholesky(K)
        elif X is not None: # then compute the cov
            K = np.cov(m=X, rowvar=False, aweights=observation_weights)
            self.L = np.linalg.cholesky(K)
        else:
            assert False,'must specify either X or K'
        self.N = self.L.shape[0]

        # specify parameters and constraints
        self.parameter_list = ['L']
        self.constraint_map = {'L':np.array([["fixed",]*self.N,]*self.N)} # first initialize to all being fixed
        if not fix_cross_cov:
            self.constraint_map['L'][np.tril(np.ones((self.N,self.N),dtype=bool),k=-1)] = '' # set the lower triangular to unconstrained
        self.constraint_map['L'][np.diag(np.ones(self.N,dtype=bool))] = '+ve' # set the diagonal elements to be positive


    def cov(self,x,z=None):
        """
        compute the covariance matrix which will just be a subset of self.K
        """
        assert np.all(np.in1d(x.squeeze(),np.arange(self.N), assume_unique=True)), 'x must contain only integer values in range(N)'
        if z is not None:
            assert np.all(np.in1d(z.squeeze(),np.arange(self.N), assume_unique=True)), 'z must contain only integer values in range(N)'
        x,z = self._process_cov_inputs(x,z) # process inputs
        K = self.L.dot(self.L.T)
        return K[np.int32(x),np.int32(z.squeeze())] # first index must be col vector for numpy to select a block


    def fix_variance(self):
        logger.info("fixing the first variance of the multi-output kernel")
        self.constraint_map['L'][0,0] = "fixed"


