# encoding=utf8
import abc
import logging
from scipy.stats import norm
import numpy as np

from smac.epm.base_epm import AbstractEPM

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class AbstractAcquisitionFunction(object):
    __metaclass__ = abc.ABCMeta
    long_name = ""

    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model: AbstractEPM, **kwargs):
        """
        A base class for acquisition functions.

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.

        """
        self.model = model

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the model is updated. E.g.
        Entropy search uses it to update it's approximation of P(x=x_min),
        EI uses it to update the current fmin.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, X: np.ndarray):
        """
        Computes the acquisition value for a given point X

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        """

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray):
        """
        Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()


class EI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""
        Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) :=
            \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                f_{t+1}(\mathbf{X}) - \xi\right] \} ]`, with
        :math:`f(X^+)` as the incumbent.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """
        Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m - self.par) / s
        f = (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            self.logger.warn("Predicted std is 0.0 for at least one sample.")
            f[s == 0.0] = 0.0

        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return f


class EIPS(EI):
    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""
        Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) :=
            \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) -
                  f_{t+1}(\mathbf{X}) - \xi\right] \} ]}
                  {np.log10(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                 predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(EIPS, self).__init__(model, par=par)
        self.long_name = 'Expected Improvement per Second'

    def _compute(self, X: np.ndarray, **kwargs):
        """
        Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        assert m.shape[1] == 2
        assert v.shape[1] == 2
        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The model already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        z = (self.eta - m_cost - self.par) / s
        f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
        f = f / m_runtime
        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            self.logger.warn("Predicted std is 0.0 for at least one sample.")
            f[s == 0.0] = 0.0

        if (f < 0).any():
            raise ValueError("Expected Improvement per Second is smaller than "
                             "0 for at least one sample.")

        return f.reshape((-1, 1))


class LogEI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractEPM,
                 par: float=0.0,
                 **kwargs):
        r"""
        Computes for a given x the logarithm expected improvement as
        acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A model that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(LogEI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """
        Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        f_min = self.eta - self.par
        v = (np.log(f_min) - m) / std
        log_ei = (f_min * norm.cdf(v)) - \
            (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            self.logger.warn("Predicted std is 0.0 for at least one sample.")
            log_ei[std == 0.0] = 0.0

        if (log_ei < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))
    
class WARM_EI(EI):
    def __init__(self,
                 model,
                 par=0.0,
                 warm_models=[],
                 **kwargs):
        '''
            see doc of inherit class
        '''

        super(WARM_EI, self).__init__(model=model, par=par, **kwargs)
        self.long_name = 'Warmstarted Expected Improvement'
        
        self.warm_models = warm_models
        self.mins = []
    
    def update(self, X, **kwargs):
        '''
            update estimated best y across instances;
            needed to compute transfer function
            
            Parameters
            ----------
            X: np.ndarray(N, D), The input points where the acquisition function
                should be evaluated. The dimensionality of X is (N, D), with N as
                the number of points to evaluate at and D is the number of
                dimensions of one X.
        '''
        
        super(EI, self).update(**kwargs)
        self.mins = []
        for model in self.warm_models:
            # remove instance features
            if X.shape[1] > model.n_params:
                X = X[:,:model.n_params] 
            y = model.predict_marginalized_over_instances(X)[0]
            self.mins.append(np.min(y))
        
    def transfer_func(self, X):
        '''
            transfer function 
            
            Parameters
            ----------
            X: np.ndarray(N, D), The input points where the acquisition function
                should be evaluated. The dimensionality of X is (N, D), with N as
                the number of points to evaluate at and D is the number of
                dimensions of one X.
        '''
        v = []
        idx = 0
        for min_y, model in zip(self.mins, self.warm_models):
            y = model.predict_marginalized_over_instances(X)[0]
            y = [y_[0] for y_ in y]
            y = np.min([y, [min_y]*len(y)], axis=0)
            v.append(y)
            idx += 1
        return np.mean(v,axis=0)
        
    def _compute(self, X, **kwargs):
        '''
            compute EI and add transfer function from warmstarted models
            
            Parameters
            ----------
            X: np.ndarray(N, D), The input points where the acquisition function
                should be evaluated. The dimensionality of X is (N, D), with N as
                the number of points to evaluate at and D is the number of
                dimensions of one X.
        '''
        #self.logger.debug("CALL WEI")
        ei_y = super(WARM_EI, self)._compute(X=X)
        transfer_y = self.transfer_func(X)
        transfer_y = [[t] for t in transfer_y]
        #for ey,ty in zip(ei_y,transfer_y):
            #self.logger.debug("EI: %f + Transfer: %f" %(ey[0],ty[0]))
        # since we maximize ACQ, but minimized transfer_y,
        # we have to ei + -1* transfer
        return ei_y - transfer_y
