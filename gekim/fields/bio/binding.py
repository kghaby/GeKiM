import numpy as np
from lmfit import Parameters as lmfitParameters
from lmfit.model import ModelResult
from typing import Union

from ...utils.fitting import general_fit, merge_params

#TODO: normalize for fit for x axis as well

def res_time():
    raise NotImplementedError()

def dose_response(dose: np.ndarray,Khalf: float,n=1,uplim=1,nonspecific_m=0): 
    '''
    Calculates the Hill equation for a dose-response curve.

    Parameters
    ----------
    dose : np.ndarray
        Array of input concentrations of the ligand.
    Khalf : float
        The dose required for half output response.
    n : float, optional
        Hill coefficient, default is 1.
    uplim : float, optional
        The upper limit of the response, default is 1, ie 100%.
    nonspecific_m : float, optional
        The slope of the nonspecific term, default is 0.

    Returns
    -------
    np.ndarray
        The fraction of the responding population.
    '''
    return uplim / (1+(Khalf/dose)**n) + nonspecific_m*dose

def dose_response_fit(dose: np.ndarray, response: np.ndarray, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    """
    Fit parameters (Khalf, kinact, n) to response with respect to dose using 
    a structured dictionary for parameters.

    Parameters
    ----------
    dose : np.ndarray
        Array of input concentrations of the ligand.
    response : np.ndarray
        Array of the fraction of the responding population.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('Khalf', value=100, vary=True, min=0, max=np.inf)
        default_params.add('n', value=1, vary=True, min=0, max=np.inf)
        default_params.add('uplim', value=1, vary=True)
        default_params.add('nonspecific_m', value=0, vary=False)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "n": {"vary": False},    
        }
        ```
    xlim : tuple, optional
        Limits for the time points considered in the fit (min_t, max_t).
    weights_kde : bool, optional
        If True, calculate the density of the x-values and use the normalized reciprocol as weights. Similar to 1/sigma for scipy.curve_fit.
        Helps distribute weight over unevenly-spaced points. Default is False.
    weights : np.ndarray, optional
        weights parameter for fitting. This argument is overridden if weights_kde=True. Default is None.
    verbosity : int, optional
        0: print nothing. 1: print upon bad fit. 2: print always. Default is 2.
    kwargs : dict, optional
        Additional keyword arguments to pass to the lmfit Model.fit function.

    Returns
    -------
    lmfit.ModelResult
        The result of the fitting operation from lmfit.
    """
    default_params = lmfitParameters()
    default_params.add('Khalf', value=100, vary=True, min=0, max=np.inf)
    default_params.add('n', value=1, vary=True, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=True)
    default_params.add('nonspecific_m', value=0, vary=False)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(dose_response, dose, response, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

class Params:
    """
    Common place for parameters found in general binding Khalfnetics literature.
    """
    @staticmethod
    def Kd(kon, koff):
        """
        Kd (i.e. dissociation constant) calculation
        
        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1)
        koff : float
            Off-rate constant (TIME^-1)
        
        Returns
        -------
        float
            The calculated dissociation constant (Kd)
        """
        return koff / kon
    
    @staticmethod
    def Keq(kon, koff):
        """
        Keq (i.e. equilibrium constant) calculation
        
        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1)
        koff : float
            Off-rate constant (TIME^-1)
        
        Returns
        -------
        float
            The calculated equilibrium constant (Keq)
        """
        return kon / koff

    
