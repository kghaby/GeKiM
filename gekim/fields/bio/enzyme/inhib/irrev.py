import numpy as np
from lmfit import Parameters as lmfitParameters
from lmfit.model import ModelResult
from typing import Union
from multiprocessing import Pool, cpu_count, Queue, Process

from .....systems.system import System
from .....simulators.ode_solver import ODESolver
from .....utils.helpers import update_dict, chunks, printer, CaptureOutput
from .....utils.fitting import general_fit, merge_params
#from .....utils.experiments import ExperimentResult

#TODO: fit to scheme. meaning yuo make a scheme without values for the transitions and fit it to occ data to see what values of rates satisfy curve
#TODO: class-based fittings will prob be better so that i can have baseclasses, reducing lots of reptition (esp docstrings) 
#TODO: reorganize? There are starting to be lots of functions

def occ_final_wrt_t(t, kobs, Etot, uplim=1, kns=0, concI0=10) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to time.

    Parameters
    ----------
    t : np.ndarray
        Array of timepoints.
    kobs : float
        Observed rate constant.
    Etot : float
        Total concentration of E across all species.
    uplim : float, optional
        Upper limit scalar of the curve. The fraction of total E typically. Default is 1, i.e., 100%.
    kns : float, optional
        Non-specific binding rate constant. Default is 0. 
    concI0 : float, optional
        Initial concentration of the ligand. Default is 10. Used for nonspecific term.

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
        
    Notes
    -----
    The nonspecific term is linear and not does model ligand consumption. 
    
    For high ligand concentrations, the nonspecific term will dominate. 
    
    Careful that the ligand concentration is not significantly diminished by nonspecific binding during the time course, if you want a well-behaved fit.
    '''
    return uplim * Etot * (1 - np.e**(-kobs * t)) + Etot * concI0 * kns * t

def kobs_uplim_fit_to_occ_final_wrt_t(t, occ_final, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    t : np.ndarray
        Array of timepoints.
    occ_final : np.ndarray
        Array of observed occupancy, i.e. concentration.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        # Observed rate constant
        default_params.add('kobs', value=0.01, vary=True, min=0, max=np.inf)
        # Total concentration of E over all species
        default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
        # Scales the upper limit of the curve
        default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)
        # Non-specific binding rate constant
        default_params.add('kns', value=0, vary=False, min=0, max=np.inf)
        # Initial concentration of the ligand (for nonspecific term. Can be ignored if kns=0)
        default_params.add('concI0', value=10, vary=False, min=0, max=np.inf)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('kobs', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)
    default_params.add('kns', value=0, vary=False, min=0, max=np.inf)
    default_params.add('concI0', value=10, vary=False, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)

    return general_fit(occ_final_wrt_t, t, occ_final, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)




def occ_total_wrt_t(t, kobs, concI0, KI, Etot, uplim=1, kns=0) -> np.ndarray:
    '''
    Calculates pseudo-first-order total occupancy of all bound states, 
    assuming fast reversible binding equilibrated at t=0.

    Parameters
    ----------
    t : np.ndarray
        Array of timepoints.
    kobs : float
        Observed rate constant.
    concI0 : float
        Initial concentration of the (saturating) inhibitor.
    KI : float
        Inhibition constant, where kobs = kinact/2, analogous to K_M, K_D, and K_A. 
        Must be in the same units as concI0.
    Etot : float
        Total concentration of E across all species.
    uplim : float, optional
        Upper limit scalar of the curve. The fraction of total E typically. Default is 1, i.e., 100%.
    kns : float, optional
        Non-specific binding rate constant. Default is 0. 
    Returns
    -------
    np.ndarray
        Occupancy of total occupancy (Occ_tot).
    '''

    FO = 1 / (1 + (KI / concI0)) # Equilibrium occupancy of reversible portion
    return uplim * Etot * (1 - (1 - FO) * np.exp(-kobs * t)) + Etot * concI0 * kns * t

def kobs_KI_uplim_fit_to_occ_total_wrt_t(t: np.ndarray, occ_tot: np.ndarray, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs and KI to the total occupancy of all bound states over time, 
    assuming fast reversible binding equilibrated at t=0.

    Parameters
    ----------
    t : np.ndarray
        Array of timepoints.
    occ_tot : np.ndarray
        Array of total bound enzyme population. All occupied states.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        # Observed rate constant
        default_params.add('kobs', value=0.01, vary=True, min=0, max=np.inf)
        # Initial concentration of the (saturating) inhibitor
        default_params.add('concI0', value=100, vary=True, min=0, max=np.inf)
        # Inhibition constant where kobs = kinact/2.
        default_params.add('KI', value=10, vary=True, min=0, max=np.inf)
        # Total concentration of E across all species
        default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
        # Scales the upper limit of the curve
        default_params.add('uplim', value=1, vary=True, min=0, max=np.inf)    
        # Non-specific binding rate constant
        default_params.add('kns', value=0, vary=False, min=0, max=np.inf)  
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''
    default_params = lmfitParameters()
    default_params.add('kobs', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('concI0', value=100, vary=True, min=0, max=np.inf)
    default_params.add('KI', value=10, vary=True, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=True, min=0, max=np.inf)
    default_params.add('kns', value=0, vary=False, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(occ_total_wrt_t, t, occ_tot, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def kobs_wrt_concI0(concI0, KI, kinact, n=1): 
    '''
    Calculates the observed rate constant kobs with respect to the initial 
    concentration of the inhibitor using a Michaelis-Menten-like equation.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor.
    KI : float
        Inhibition constant, analogous to K_M, K_D, and K_A, where kobs = kinact/2.
    kinact : float
        Maximum potential rate of covalent bond formation.
    n : float, optional
        Hill coefficient, default is 1.

    Returns
    -------
    np.ndarray
        Array of kobs values, the first order observed rate constants of inactivation, 
        with units of inverse time.
    
    Notes
    -----
    Assumes that concI is constant over the timecourses where kobs is calculated. 
    '''
    return kinact / (1 + (KI / concI0)**n)

def KI_kinact_n_fit_to_kobs_wrt_concI0(concI0: np.ndarray, kobs: np.ndarray, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    """
    Fit parameters (KI, kinact, n) to kobs with respect to concI0 using 
    a structured dictionary for parameters.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor.
    kobs : np.ndarray
        Array of observed rate constants.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
        default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
        default_params.add('n', value=1, vary=False, min=0, max=np.inf)   
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "n": {"vary": True},    
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
    
    Assumes that concI is constant over the timecourses where kobs is calculated. 
    """
    default_params = lmfitParameters()
    default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('n', value=1, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(kobs_wrt_concI0, concI0, kobs, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def kobs_wrt_concI0_1step(concI0, Eff, y_ctrl=0): 
    '''
    Calculates the observed rate constant kobs with respect to the initial 
    concentration of the inhibitor using the second-order efficiency constant, kinact/KI.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor.
    Eff : float
        Inactivation efficiency (ie covalent efficiency)
    y_ctrl : float, optional
        Control value for the y-intercept. Default is 0.

    Returns
    -------
    np.ndarray
        Array of kobs values, the first order observed rate constants of inactivation, 
        with units of inverse time.
    
    Notes
    -----
    Assumes that concI is constant over the timecourses where kobs is calculated. 
    '''
    return Eff * concI0 + y_ctrl

def Eff_fit_to_kobs_wrt_concI0(concI0: np.ndarray, kobs: np.ndarray, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    """
    Fit parameters (KI, kinact, n) to kobs with respect to concI0 using 
    a structured dictionary for parameters.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor.
    kobs : np.ndarray
        Array of observed rate constants.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('Eff', value=1e-4, vary=True, min=0, max=np.inf)
        default_params.add('y_ctrl', value=0, vary=False, min=0, max=np.inf)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "n": {"vary": True},    
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
    
    Assumes that concI is constant over the timecourses where kobs is calculated. 
    """
    default_params = lmfitParameters()
    default_params.add('Eff', value=1e-4, vary=True, min=0, max=np.inf)
    default_params.add('y_ctrl', value=0, vary=False, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(kobs_wrt_concI0_1step, concI0, kobs, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def dose_response(dose: np.ndarray,Khalf: float, kinact: float, t: float, n=1): 
    '''
    Calculates the Hill equation for a dose-response curve.

    Parameters
    ----------
    dose : np.ndarray
        Array of input concentrations of the ligand.
    Khalf : float
        The dose required for half output response.
    t : float
        The endpoint used of dosing. 
    kinact : 
        The apparent maximal rate constant of inactivation. 
    n : float, optional
        Hill coefficient, default is 1.

    Returns
    -------
    np.ndarray
        The fraction of the responding population.
    '''
    return (1 - np.exp(-kinact * t)) / (1 + (Khalf / dose)**n)

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
        default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
        default_params.add('t', value=3600, vary=False)
        default_params.add('n', value=1, vary=True, min=0, max=np.inf)
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
    default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('t', value=3600, vary=False)
    default_params.add('n', value=1, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(dose_response, dose, response, lm_params, xlim=xlim, weights_kde=weights_kde, 
                        weights=weights, verbosity=verbosity, **kwargs)

def occ_final_wrt_concI0(concI0: np.ndarray, t: float, KI: float, kinact: float,  n: float = 1, Etot: float = 1, 
                        uplim: float = 1) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to dose.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    t : float
        Endpoint for the dosing.
    KI : float
        Inhibition constant, analogous to K_M.
    kinact : float
        Maximum potential rate of covalent bond formation.
    Etot : float
        Total concentration of E across all species. Default is 1. 
        Leave as 1 to have the function return the fraction of total E.
    uplim : float, optional
        Upper limit scalar of the curve. The fraction of total E typically. Default is 1, i.e., 100%.

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
    '''
    kobs = kobs_wrt_concI0(concI0,KI,kinact,n=n)
    return uplim * Etot * (1 - np.e**(-kobs * t))

def fit_to_occ_final_wrt_concI0(concI0, occ_final, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    occ_final : np.ndarray
        Array of observed occupancy, i.e. concentration.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
        default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
        default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
        default_params.add('n', value=1, vary=False, min=0, max=np.inf)
        default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
        default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
    default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('n', value=1, vary=False, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)


    lm_params = merge_params(default_params, nondefault_params)

    return general_fit(occ_final_wrt_concI0, concI0, occ_final, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def occ_final_wrt_concI0_norm(concI0: np.ndarray, t: float, KI: float, kinact: float,  n: float = 1, Etot: float = 1, 
                        uplim: float = 1) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to dose.
    Assumes that data is normalized to the max response.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    t : float
        Endpoint for the dosing.
    KI : float
        Inhibition constant, analogous to K_M.
    kinact : float
        Maximum potential rate of covalent bond formation.
    Etot : float
        Total concentration of E across all species. Default is 1. 
        Leave as 1 to have the function return the fraction of total E.
    uplim : float, optional
        Upper limit scalar of the curve. The fraction of total E typically. Default is 1, i.e., 100%.

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
    '''
    kobs = kobs_wrt_concI0(concI0,KI,kinact,n=n)
    return uplim * Etot * (1 - np.e**(-kobs * t))/(1 - np.e**(-kinact * t))

def fit_to_occ_final_wrt_concI0_norm(concI0, occ_final, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    occ_final : np.ndarray
        Array of observed occupancy, i.e. concentration.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
        default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
        default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
        default_params.add('n', value=1, vary=False, min=0, max=np.inf)
        default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
        default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
    default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('n', value=1, vary=False, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)


    lm_params = merge_params(default_params, nondefault_params)

    return general_fit(occ_final_wrt_concI0_norm, concI0, occ_final, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def occ_final_wrt_concI0_4S(concI0: np.ndarray, t: float, kon: float, koff: float, kinact: float,  Parm: float, n: float = 1, Etot: float = 1, 
                        uplim: float = 1) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to dose.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    t : float
        Endpoint for the dosing.
    KI : float
        Inhibition constant, analogous to K_M.
    kinact : float
        Maximum potential rate of covalent bond formation.
    Etot : float
        Total concentration of E across all species. Default is 1. 
        Leave as 1 to have the function return the fraction of total E.
    uplim : float, optional
        Upper limit scalar of the curve. The fraction of total E typically. Default is 1, i.e., 100%.

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
    '''
    kinact_app = kinact*Parm
    koff_app = koff*(1-Parm)
    KI_app = Params.KI(kon, koff_app, kinact_app)

    kobs = kobs_wrt_concI0(concI0,KI_app,kinact_app,n=n)
    return uplim * Etot * (1 - np.e**(-kobs * t))

def fit_to_occ_final_wrt_concI0_4S(concI0, occ_final, nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, 
                                        weights_kde=False, weights: np.ndarray = None, verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    occ_final : np.ndarray
        Array of observed occupancy, i.e. concentration.
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
        default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
        default_params.add('koff', value=.01, vary=True, min=0, max=np.inf)
        default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf)
        default_params.add('Parm', value=0.5, vary=False, min=0, max=1)
        default_params.add('n', value=1, vary=False, min=0, max=np.inf)
        default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
        default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
    default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
    default_params.add('koff', value=.01, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf)
    default_params.add('Parm', value=0.5, vary=False, min=0, max=1)
    default_params.add('n', value=1, vary=False, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=False, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)

    return general_fit(occ_final_wrt_concI0_4S, concI0, occ_final, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def dose_response_from_odes(concI0_arr: np.ndarray, t: float, kon: float, koff: float, kinact: float, 
        scheme: dict, experiment_kwargs: dict = None) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to dose.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    t : float
        Endpoint for the dosing.
    kon : float
        
    koff : float

    kinact : float
        Maximum potential rate of covalent bond formation.
        
    scheme : dict

    experiment_kwargs : dict
        kwargs for Experiments.dose_response

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
    '''
    kinact_app = kinact
    koff_app = koff
    k_changes = {
        'kon': kon,
        'koff': koff_app,
        'kinact': kinact_app
    }

    if experiment_kwargs is None:
        experiment_kwargs = {}

    system, response = Experiments.dose_response(concI0_arr, t, scheme, k_changes=k_changes, **experiment_kwargs)

    return response

def fit_dose_response_from_odes(concI0_arr: np.ndarray, response_obs: np.ndarray, scheme: dict,  experiment_kwargs: dict = None, 
        nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, weights_kde=False, weights: np.ndarray = None, 
        verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    response_obs : np.ndarray
        Array of observed occupancy, i.e. concentration.
    scheme : dict
        The scheme of the system.
    experiment_kwargs : dict
        kwargs for Experiments.dose_response
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
        default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
        default_params.add('koff', value=.01, vary=True, min=0, max=np.inf) # koff_app
        default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf) # kinact_app

        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
    default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
    default_params.add('koff', value=.01, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)

    model_kwargs = {
        "scheme": scheme,
        "experiment_kwargs": experiment_kwargs
    }

    return general_fit(dose_response_from_odes, concI0_arr, response_obs, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, model_kwargs=model_kwargs, **kwargs)
    
def dose_response_from_odes_4S(concI0_arr: np.ndarray, t: float, kon: float, koff: float, kinact: float,  Parm: float, 
        scheme: dict, experiment_kwargs: dict = None) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to dose.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    t : float
        Endpoint for the dosing.
    kon : float
        
    koff : float

    kinact : float
        Maximum potential rate of covalent bond formation.

    Parm : float
        
    scheme : dict

    experiment_kwargs : dict
        kwargs for Experiments.dose_response

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
    '''
    kinact_app = kinact*Parm
    koff_app = koff*(1-Parm)
    k_changes = {
        'kon': kon,
        'koff': koff_app,
        'kinact': kinact_app
    }
    
    if experiment_kwargs is None:
        experiment_kwargs = {}

    system, response = Experiments.dose_response(concI0_arr, t, scheme, k_changes=k_changes, **experiment_kwargs)

    return response

def fit_dose_response_from_odes_4S(concI0_arr: np.ndarray, response_obs: np.ndarray, scheme: dict,  experiment_kwargs: dict = None, 
        nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, weights_kde=False, weights: np.ndarray = None, 
        verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor. Note that the 
        inhibitor is assumed to be constant so that the initial concentration 
        may be used in place of the remaining [I] at the endpoint.
    response_obs : np.ndarray
        Array of observed occupancy, i.e. concentration.
    scheme : dict
        The scheme of the system.
    experiment_kwargs : dict
        kwargs for Experiments.dose_response
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
        default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
        default_params.add('koff', value=.01, vary=True, min=0, max=np.inf)
        default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf)
        default_params.add('Parm', value=0.5, vary=False, min=0, max=1)
        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('t', value=3600, vary=False, min=0, max=np.inf)
    default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
    default_params.add('koff', value=.01, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf)
    default_params.add('Parm', value=0.5, vary=False, min=0, max=1)

    lm_params = merge_params(default_params, nondefault_params)

    model_kwargs = {
        "scheme": scheme,
        "experiment_kwargs": experiment_kwargs
    }

    return general_fit(dose_response_from_odes_4S, concI0_arr, response_obs, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, model_kwargs=model_kwargs, **kwargs)

def time_response_from_odes(t_eval: np.ndarray, kon: float, koff: float, kinact: float, scheme: dict, experiment_kwargs: dict = None) -> np.ndarray:
    '''
    Calculate the occupancy of final occupancy (Occ_cov) with respect to dose.

    Parameters
    ----------
    t_eval : np.ndarray

    experiment_kwargs : dict
        kwargs for Experiments.time_response_nofit
    
    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).

    '''
    kinact_app = kinact
    koff_app = koff
    k_changes = {
        'kon': kon,
        'koff': koff_app,
        'kinact': kinact_app
    }
    
    if experiment_kwargs is None:
        experiment_kwargs = {}

    experiment_kwargs["sim_kwargs"] = {
        't_eval': t_eval
    }

    system, response = Experiments.time_response_nofit(scheme, k_changes=k_changes, **experiment_kwargs)

    return response

def fit_time_response_from_odes(t_eval: np.ndarray, response_obs: np.ndarray, scheme: dict, experiment_kwargs: dict = None, 
        nondefault_params: Union[dict,lmfitParameters] = None, xlim: tuple = None, weights_kde=False, weights: np.ndarray = None, 
        verbosity=2, **kwargs) -> ModelResult:
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    t_eval : np.ndarray

    response_obs : np.ndarray
        Array of observed occupancy, i.e. concentration.
    scheme : dict
        The scheme of the system.
    experiment_kwargs : dict
        kwargs for Experiments.time_response_nofit
    nondefault_params : dict or Parameters, optional
        A structured dictionary of parameters with 'value','vary', and 'bound' keys or a lmfit.Parameters object.
        Defaults:
        ```python
        default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
        default_params.add('koff', value=.01, vary=True, min=0, max=np.inf) # koff_app
        default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf) # kinact_app

        ```
        Example dict of nondefaults:
        ```python
        nondefault_params = {
            "Etot": {"vary": False, "value": 0.5},  
            "uplim": {"vary": False},    
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

    '''

    default_params = lmfitParameters()
    default_params.add('kon', value=.0001, vary=True, min=0, max=np.inf)
    default_params.add('koff', value=.01, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.001, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)

    model_kwargs = {
        "scheme": scheme,
        "experiment_kwargs": experiment_kwargs
    }

    return general_fit(time_response_from_odes, t_eval, response_obs, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, model_kwargs=model_kwargs, **kwargs)

    
class Params:
    """
    Common place for parameters found in covalent inhibition literature.
    """
    #TODO: rename Constants
    @staticmethod
    def Ki(kon, koff):
        """
        Ki (i.e. inhib. dissociation constant, Kd) calculation

        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1).
        koff : float
            Off-rate constant (TIME^-1).

        Returns
        -------
        float
            The calculated inhib. dissociation constant (Ki).

        Notes
        -----
        The inhib. dissociation constant (Ki) is calculated as koff / kon.
        """
        return koff / kon

    @staticmethod
    def KI(kon, koff, kinact):
        """
        KI (i.e. inhibition constant, KM, KA, Khalf, KD (not to be confused with Kd)) calculation.
        Numerically, this should be the concentration of inhibitor that yields kobs = 1/2*kinact.

        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1).
        koff : float
            Off-rate constant (TIME^-1).
        kinact : float
            Inactivation (last irreversible step) rate constant.

        Returns
        -------
        float
            The calculated inhibition constant (KI).

        Notes
        -----
        The inhibition constant (KI) is calculated as (koff + kinact) / kon.
        """
        return (koff + kinact) / kon

    @staticmethod
    def kinact_app_from_uplim(t: float, Prob_cov: float):
        """
        The apparent maximal rate constant of inactivation, calculated from a single timepoint.

        Parameters
        ----------
        t : float
            Endpoint for the dosing.
        Prob_cov : float
            Probability (ie fraction) of the covalently-bound state.

        Notes
        -----
        The data must be from a timepoint where kobs ~= kinact due to large [I].

        Can be obtained from the upper limit of a fitted dose response curve. 
        """
        if Prob_cov >= 1:
            print("Cannot calculate kinact_app from a timepoint where the system is fully inactivated.")
            return np.inf
        return -np.log(1-Prob_cov)/t
    
    @staticmethod
    def CE(kon, koff, kinact):
        """
        Covalent efficiency (i.e. specificity, potency) calculation (kinact/KI).

        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1).
        koff : float
            Off-rate constant (TIME^-1).
        kinact : float
            Inactivation (last irreversible step) rate constant.

        Returns
        -------
        float
            The calculated covalent efficiency (kinact/KI).

        Notes
        -----
        The covalent efficiency is calculated as the ratio of 
        the inactivation rate constant (kinact) to the inhibition constant (KI).
        """
        return kinact/Params.KI(kon, koff, kinact)

    def Khalf_from_KI_kinact(KI_app, kinact_app, t):
        """
        Calculate the Khalf (ie EC50) from the apparent KI and kinact and a single timepoint.
        """
        log_term = np.log((np.e**(-kinact_app*t)+1)/2)
        Khalf = -1*((KI_app * log_term) / (kinact_app  * t + log_term))
        return Khalf

    def KI_from_Khalf_kinact(Khalf,kinact_app,t):
        """
        Calculate the KI from the Khalf (EC50) and apparent kinact and a single timepoint.
        """
        log_term = np.log((np.e**(-kinact_app*t)+1)/2)
        KI = Khalf*(-1 * kinact_app * t / (log_term) - 1)
        return KI

class Experiments:
    """
    Common place for experimental setups in covalent inhibition literature.
    """
    #TODO: make base class for experiments. have the docstrings be inherited. 
    #TODO: make class for experiment output
    
    @staticmethod
    def time_response_nofit(scheme: Union[dict,System], t_eval=None, dose_spname: str = "I", CO_spname: str = "EI", E_spname: str = "E", response_sp: list = None, 
                    k_changes: dict = None, system_kwargs: dict = None, sim_kwargs: dict = None) -> tuple[System,np.ndarray]:
        """
        A macro for doing timecourses.

        Returns
        -------
        System
            The system object.
        np.ndarray
            The response normalized to the total enzyme.

        Notes
        -----

        Example k_changes dict:
        ```python
        k_changes = {
            'kon': kon,
            'koff': koff, # koff_app
            'kinact': kinact # kinact_app
        }
        ```
        """
        # Default to CO_spname for the response species
        if response_sp is None:
            response_sp = [CO_spname]

        if isinstance(scheme, dict):
            default_system_kwargs = {
                "quiet": True,
            }
            system_kwargs = update_dict(default_system_kwargs, system_kwargs)
            system = System(scheme,**system_kwargs)
        elif isinstance(scheme, System):
            system = scheme
        else:
            raise ValueError("scheme must be a dict or System object.")

        default_sim_kwargs = {
            "t_eval": t_eval
        }
        sim_kwargs = update_dict(default_sim_kwargs, sim_kwargs)

        if k_changes is not None:
            for k_name, k_val in k_changes.items():
                if k_name not in system.transitions:
                    raise ValueError(f"{k_name} not found in system transitions.")
                system.transitions[k_name].k = k_val
        system.simulator = ODESolver(system)
        system.simulator.simulate(**sim_kwargs)
        response = system.sum_species_simout(whitelist=response_sp) 
        response /= system.species[E_spname].y0 # normalize to total E

        return system, response

    @staticmethod
    def time_response(scheme: Union[dict,System], t_eval=None, dose_spname: str = "I", CO_spname: str = "EI", E_spname: str = "E", 
                    k_changes: dict = None, system_kwargs: dict = None, sim_kwargs: dict = None, 
                    fit_occ_kwargs: dict = None) -> tuple[System, ModelResult]:
        """
        A macro for doing timecourses.

        Returns
        -------
        System
            The system object.
        ModelResult
            The fit output.

        Notes
        -----

        Example k_changes dict:
        ```python
        k_changes = {
            'kon': kon,
            'koff': koff, # koff_app
            'kinact': kinact # kinact_app
        }
        ```
        """

        if isinstance(scheme, dict):
            concE0 = scheme["species"][E_spname]["y0"]
        elif isinstance(scheme, System):
            concE0 = scheme.species[E_spname].y0
        else:
            raise ValueError("scheme must be a dict or System object.")

        default_fit_occ_kwargs = {
            "nondefault_params" : {
                    "Etot": {"vary": False, "value": concE0},
                    "uplim": {"vary": False, "value": 1},
                },
            "verbosity": 2,
        }
        fit_occ_kwargs = update_dict(default_fit_occ_kwargs, fit_occ_kwargs)

        system, response = Experiments.time_response_nofit(scheme,t_eval=t_eval,dose_spname=dose_spname,CO_spname=CO_spname,E_spname=E_spname, k_changes=k_changes, system_kwargs=system_kwargs,sim_kwargs=sim_kwargs)
    
        x_data = system.simout["t"]
        y_data = response
        fit_output = kobs_uplim_fit_to_occ_final_wrt_t(x_data,y_data,**fit_occ_kwargs)
        fit_output.best_values["kobs"]

        return system, fit_output
    

    @staticmethod
    def _single_dose_for_rate(args):
        indices, doses, scheme, dose_spname, CO_spname, k_changes, system_kwargs, sim_kwargs, fit_occ_kwargs = args

        with CaptureOutput() as output:
            system = System(scheme,**system_kwargs)
            system.species[dose_spname].y0 = doses
            if k_changes is not None:
                for k_name, k_val in k_changes.items():
                    if k_name not in system.transitions:
                        raise ValueError(f"{k_name} not found in scheme transitions.")
                    system.transitions[k_name].k = k_val
            system.simulator = ODESolver(system)
            system.simulator.simulate(**sim_kwargs)

            kobs_arr_mini = np.zeros_like(doses)
            for i,dose in enumerate(doses):
                # The type of the sim output depends on the size of the initial concentrations. 
                # Will be a list of arrays if len(doses) > 1
                if len(doses) > 1:
                    x_data = system.simout["t"][i]
                    y_data = system.species[CO_spname].simout["y"][i]
                else:
                    x_data = system.simout["t"]
                    y_data = system.species[CO_spname].simout["y"]
                
                fit_output = kobs_uplim_fit_to_occ_final_wrt_t(x_data,y_data,**fit_occ_kwargs)
                kobs_arr_mini[i] = fit_output.best_values["kobs"]

        return indices, kobs_arr_mini, list(output)

    @staticmethod
    def dose_rate(scheme: dict, dose_arr: np.ndarray, dose_spname: str = "I", CO_spname: str = "EI", E_spname: str = "E", num_cores=1, 
                k_changes: dict = None, system_kwargs: dict = None, sim_kwargs: dict = None, fit_occ_kwargs: dict = None, 
                fit_kobs_kwargs: dict = None) -> ModelResult:
        """
        A macro for doing timecourses with variable [I] and fitting for apparent KI and kinact.
        Uses multiprocessing.

        Notes
        -----
        Different processes may have different atols if atol0=0 and dose_arr ranges 
            from being smaller to larger than `scheme["species"][E_spname]["y0"]`
        This is because if atol==0 is chosen in `simulate()` to be `1e-6*min(y0[y0!=0])`

        Example k_changes dict:
        ```python
        k_changes = {
            'kon': kon,
            'koff': koff, # koff_app
            'kinact': kinact # kinact_app
        }
        ```
        """
        num_cores = min(num_cores, cpu_count())

        default_system_kwargs = {
            "quiet": True,
        }
        system_kwargs = update_dict(default_system_kwargs, system_kwargs)

        default_sim_kwargs = {
        }
        sim_kwargs = update_dict(default_sim_kwargs, sim_kwargs)

        default_fit_occ_kwargs = {
            "nondefault_params" : {
                    "Etot": {"vary": False, "value": scheme["species"][E_spname]["y0"]},
                    "uplim": {"vary": False, "value": 1},
                },
            "verbosity": 1,
        }
        fit_occ_kwargs = update_dict(default_fit_occ_kwargs, fit_occ_kwargs)

        default_fit_kobs_kwargs = {
            "nondefault_params" : {
                    "n": {"vary": False}
                    },
            "verbosity": 2,
        }
        fit_kobs_kwargs = update_dict(default_fit_kobs_kwargs, fit_kobs_kwargs)
            

        # This is a multiprocessing queue to prevent overlapping prints from different processes
        queue = Queue()
        printer_process = Process(target=printer, args=(queue,))
        printer_process.start()
                     
        # Set up process inputs
        chunked_indices = list(chunks(range(len(dose_arr)), len(dose_arr) // num_cores + 1))
        chunked_doses = list(chunks(dose_arr, len(dose_arr) // num_cores + 1))

        args_list = [(indices, doses, scheme.copy(), dose_spname, CO_spname, k_changes, system_kwargs, sim_kwargs, fit_occ_kwargs)
                     for indices, doses in zip(chunked_indices, chunked_doses)]


        # Run the simulations in parallel
        with Pool(num_cores) as pool:
            results = pool.map(Experiments._single_dose_for_rate, args_list)

        kobs_arr = np.zeros_like(dose_arr) 
        for indices, kobs_chunk, output in results:
            kobs_arr[indices[0]:indices[0] + len(kobs_chunk)] = kobs_chunk
            for line in output:
                queue.put(line)

        queue.put("DONE")
        printer_process.join()

        fit_output = KI_kinact_n_fit_to_kobs_wrt_concI0(dose_arr, kobs_arr, **fit_kobs_kwargs)

        return fit_output

    @staticmethod
    def dose_response(dose_arr: np.ndarray, t_end: float, scheme: Union[dict,System], dose_spname: str = "I", CO_spname: str = "EI", E_spname: str = "E", response_sp: list = None, k_changes: dict = None, system_kwargs: dict = None, sim_kwargs: dict = None) -> tuple[System, np.ndarray]:
        """
        A macro for doing timecourses with variable [I] and returning a fractional response curve.

        Returns
        -------
        System
            The system object.
        np.ndarray
            The response normalized to the total enzyme.

        Notes
        -----

        Example k_changes dict:
        ```python
        k_changes = {
            'kon': kon,
            'koff': koff, # koff_app
            'kinact': kinact # kinact_app
        }
        ```

        """
        # Default to CO_spname for the response species
        if response_sp is None:
            response_sp = [CO_spname]

        if isinstance(scheme, dict):
            default_system_kwargs = {
                "quiet": True,
            }
            system_kwargs = update_dict(default_system_kwargs, system_kwargs)
            system = System(scheme,**system_kwargs)
        elif isinstance(scheme, System):
            system = scheme
        else:
            raise ValueError("scheme must be a dict or System object.")

        default_sim_kwargs = {
            "t_span": (0, t_end),
        }
        sim_kwargs = update_dict(default_sim_kwargs, sim_kwargs)

        system.species[dose_spname].y0 = dose_arr
        if k_changes is not None:
            for k_name, k_val in k_changes.items():
                if k_name not in system.transitions:
                    raise ValueError(f"{k_name} not found in scheme transitions.")
                system.transitions[k_name].k = k_val
        system.simulator = ODESolver(system)
        system.simulator.simulate(**sim_kwargs)

        response = system.sum_species_simout(whitelist=response_sp)
        response = np.array([simout[-1] for simout in response])
        response /= system.species[E_spname].y0 # normalize to total E

        return system, response