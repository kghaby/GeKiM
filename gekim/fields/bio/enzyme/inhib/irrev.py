import numpy as np
from lmfit import Parameters
from lmfit.model import ModelResult
from typing import Union
from multiprocessing import Pool, cpu_count
from .....schemes import NState
from .....simulators import ODESolver
from .....utils.helpers import update_dict
from .....utils.fitting import general_fit, merge_params
#from .....utils.experiments import ExperimentResult

#TODO: fit to scheme. meaning yuo make a scheme without values for the transitions and fit it to occ data to see what values of rates satisfy curve

def occ_final_wrt_t(t,kobs,Etot,uplim=1) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        Occupancy of final occupancy (Occ_cov).
    '''
    return uplim * Etot * (1 - np.e**(-kobs * t))

def kobs_uplim_fit_to_occ_final_wrt_t(t, occ_final, nondefault_params: Union[dict,Parameters] = None, xlim: tuple = None, 
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
        default_params.add('uplim', value=1, vary=True, min=0, max=np.inf)
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

    default_params = Parameters()
    default_params.add('kobs', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)

    return general_fit(occ_final_wrt_t, t, occ_final, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)




def occ_total_wrt_t(t,kobs,concI0,KI,Etot,uplim=1):
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

    Returns
    -------
    np.ndarray
        Occupancy of total occupancy (Occ_tot).
    '''

    FO = 1 / (1 + (KI / concI0)) # Equilibrium occupancy of reversible portion
    return uplim * Etot * (1 - (1 - FO) * np.exp(-kobs * t))

def kobs_KI_uplim_fit_to_occ_total_wrt_t(t: np.ndarray, occ_tot: np.ndarray, nondefault_params: Union[dict,Parameters] = None, xlim: tuple = None, 
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
    default_params = Parameters()
    default_params.add('kobs', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('concI0', value=100, vary=True, min=0, max=np.inf)
    default_params.add('KI', value=10, vary=True, min=0, max=np.inf)
    default_params.add('Etot', value=1, vary=False, min=0, max=np.inf)
    default_params.add('uplim', value=1, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(occ_total_wrt_t, t, occ_tot, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

def kobs_wrt_concI0(concI0,KI,kinact,n=1): 
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

def KI_kinact_n_fit_to_kobs_wrt_concI0(concI0: np.ndarray, kobs: np.ndarray, nondefault_params: Union[dict,Parameters] = None, xlim: tuple = None, 
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
    default_params = Parameters()
    default_params.add('KI', value=100, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('n', value=1, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(kobs_wrt_concI0, concI0, kobs, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

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

def dose_response_fit(dose: np.ndarray, response: np.ndarray, nondefault_params: Union[dict,Parameters] = None, xlim: tuple = None, 
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
    default_params = Parameters()
    default_params.add('Khalf', value=100, vary=True, min=0, max=np.inf)
    default_params.add('kinact', value=0.01, vary=True, min=0, max=np.inf)
    default_params.add('t', value=3600, vary=False)
    default_params.add('n', value=1, vary=True, min=0, max=np.inf)

    lm_params = merge_params(default_params, nondefault_params)
    return general_fit(dose_response, dose, response, lm_params, xlim=xlim, weights_kde=weights_kde, weights=weights, verbosity=verbosity, **kwargs)

class Params:
    """
    Common place for parameters found in covalent inhibition literature.
    """
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
    def kinact_app(t: float, Prob_cov: float):
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

class Experiments:
    """
    Common place for experimental setups in covalent inhibition literature.
    """
    
    @staticmethod
    def timecourse(scheme: dict, sim_kwargs=None, fit_occ_kwargs=None):
        """
        A macro for doing timecourses.
        """
        raise NotImplementedError()
        
        default_sim_kwargs = {}
        sim_kwargs = update_dict(default_sim_kwargs, sim_kwargs)

        default_fit_occ_kwargs = {}
        fit_occ_kwargs = update_dict(default_fit_occ_kwargs, fit_occ_kwargs)

        system = gk.schemes.NState(scheme,quiet=True)
        system.simulator = gk.simulators.ODESolver(system)
        system.simulator.simulate(**sim_kwargs)
    
        return system
    
    @staticmethod
    def sim_dose(args):
        i, dose, scheme, dose_spname, CO_spname, sim_kwargs, fit_occ_kwargs = args
        system = NState(scheme,quiet=True)
        system.species[dose_spname].y0 = dose
        system.simulator = ODESolver(system)
        system.simulator.simulate(**sim_kwargs)

        fit_output = kobs_uplim_fit_to_occ_final_wrt_t(
                system.simout["t"],
                system.species[CO_spname].simout["y"],
                **fit_occ_kwargs)

        return i, fit_output.best_values["kobs"]

    @staticmethod
    def dose_rate(dose_arr, scheme: dict, dose_spname: str = "I", CO_spname: str = "EI", E_spname: str = "E", num_cores=1, 
                    sim_kwargs=None, fit_occ_kwargs=None, fit_kobs_kwargs=None):
        """
        A macro for doing timecourses with variable [I].
        """
        num_cores = min(num_cores, cpu_count())

        default_sim_kwargs = {}
        sim_kwargs = update_dict(default_sim_kwargs, sim_kwargs)

        default_fit_occ_kwargs = {
            "nondefault_params" : {
                    "Etot": {"vary": False, "value": scheme["species"][E_spname]["y0"]},
                    "uplim": {"vary": False, "value": 1}
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
            


        args_list = [(i, dose, scheme, dose_spname, CO_spname, sim_kwargs, fit_occ_kwargs) for i, dose in enumerate(dose_arr)]
        kobs_arr = np.zeros_like(dose_arr)
        with Pool(num_cores) as pool:
            results = pool.map(Experiments.sim_dose, args_list)

        for i, kobs in results:
            kobs_arr[i] = kobs

        fit_output = KI_kinact_n_fit_to_kobs_wrt_concI0(dose_arr, kobs_arr, **fit_kobs_kwargs)

        return fit_output