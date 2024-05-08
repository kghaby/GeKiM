import numpy as np
from scipy.optimize import curve_fit
from ..utils import _update_dict_with_subset
from . import fitting

# TODO: fit to scheme. meaning yuo make a scheme without values for the transitions and fit it to occ data to see what values of rates satisfy curve
# TODO: fitting can break with small occ values, like if using M units. Is this a limit of curve fit?
# TODO: detect trivial solutions for curve fitting, like if all values are the same, or if all values are 0, or if all values are 1.
# TODO: curve fit and defaults work best/expect normalized data, but the user may not know this. Add a check for this. Does Etot param contradict this?
    # Can convert Etot to reasonable units for curve fitting
#TODO: time arrays that are not evenly spaced will hurt curve fitting.
#TODO: accept dense output so a custom time array can be passed. this will help with starts misrepresenting the fit like for total occ not getting the right KI 

def occ_final_wrt_t(t,kobs,Etot,uplim=1):
    '''
    Args:nondefault_params
    t: Array of timepoints.
    kobs: Observed rate constant.
    Etot: Total concentration of E across all species.
    uplim: Upper limit scalar of the curve. 
            The fraction of total E typically. Default=1, ie 100%. 

    Returns:
    np.array: Occupancy of final occupancy (Occ_cov).
    '''
    return uplim*Etot*(1-np.e**(-kobs*t))

def kobs_uplim_fit_to_occ_final_wrt_t(t: np.ndarray, occ_final: np.ndarray, nondefault_params: dict = None, xlim=None): 
    '''
    Fit kobs to the first order occupancy over time.

    Args:
    t: Array of timepoints.
    occ_final: Array of observed occupancy, i.e. concentration.
    nondefault_params (dict): A structured dictionary of parameters with 'fixed','guess', and 'bound' keys. Include any params to update the default.
        default_params = {
            "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)}, # Observed rate constant
            "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},    # Total concentration of E over all species
            "uplim": {"fix": None, "guess": 1, "bounds": (0,np.inf)},   # Scales the upper limit of the curve
        }
    xlim (tuple): Limits for the time points considered in the fit (min_t, max_t).

    Returns:
    an instance of the FitOutput class
    
    Example:
    ```python
    fit_output =  ci.kobs_uplim_fit_to_occ_final_wrt_t(t,system.system.species["EI"].conc,nondefault_params={"Etot":{"fix":concE0}})
    ```
    Will fit kobs and uplim to the concentration of EI over time, fixing Etot at concE0.
    '''
    # Default
    params = {
        "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
        "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},
        "uplim": {"fix": None, "guess": 1, "bounds": (0,np.inf)}, 
    }

    if nondefault_params is not None:
        params = _update_dict_with_subset(params, nondefault_params)

    p0, bounds, param_order, fixed_params = fitting._extract_fit_info(params)

    if xlim:
        indices = (t >= xlim[0]) & (t <= xlim[1])
        t = t[indices]
        occ_final = occ_final[indices]

    def fitting_adapter(t, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return occ_final_wrt_t(t,all_params["kobs"],all_params["Etot"],uplim=all_params["uplim"])

    popt, pcov = curve_fit(fitting_adapter, t, occ_final, p0=p0, bounds=tuple(bounds))
    fitted_data = fitting_adapter(t, *popt)
    fit_output = fitting._prepare_output(popt, pcov, param_order, t, fitted_data, occ_final)

    return fit_output

def occ_total_wrt_t(t,kobs,concI0,KI,Etot,uplim=1):
    '''
    Calculates pseudo-first-order total occupancy of all bound states, assuming fast reversible binding equilibrated at t=0.

    Args:
    t: Array of timepoints.
    kobs: Observed rate constant.
    concI0: Initial concentration of the (saturating) inhibitor.
    KI: Inhibition constant, where kobs = kinact/2, analogous to K_M, K_D, and K_A. Must be in the same units as concI0.
    Etot: Total concentration of E across all species.
    uplim: Upper limit scalar of the curve. 
            The fraction of total E typically. Default=1, ie 100%. 
    
    Returns:
    np.array: Occupancy of total occupancy (Occ_tot).
    '''

    FO = 1/(1+(KI/concI0)) # Equilibrium occupancy of reversible portion
    return uplim*Etot*(1-(1-FO)*(np.e**(-kobs*t)))

def kobs_KI_uplim_fit_to_occ_total_wrt_t(t: np.ndarray, occ_tot: np.ndarray, nondefault_params: dict = None, xlim=None): 
    '''
    Fit kobs and KI to the total occupancy of all bound states over time, assuming fast reversible binding equilibrated at t=0.

    Args:
    t: Array of timepoints.
    occ_tot: Array of total occupancy.
    nondefault_params = {
        "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
        "concI0": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
        "KI": {"fix": None, "guess": 10, "bounds": (0,np.inf)},
        "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},
        "uplim": {"fix": None, "guess": 1, "bounds": (0,1)},        # Scales the upper limit of the curve
    }
    xlim (tuple): Limits for the time points considered in the fit (min_t, max_t).

    Returns:
    an instance of the FitOutput class
    '''
    # Default
    params = {
        "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
        "concI0": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
        "KI": {"fix": None, "guess": 10, "bounds": (0,np.inf)},
        "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},
        "uplim": {"fix": None, "guess": 1, "bounds": (0,1)}, 
    }

    if nondefault_params is not None:
        params = _update_dict_with_subset(params, nondefault_params)

    p0, bounds, param_order, fixed_params = fitting._extract_fit_info(params)

    if xlim:
        indices = (t >= xlim[0]) & (t <= xlim[1])
        t = t[indices]
        occ_tot = occ_tot[indices]

    def fitting_adapter(t, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return occ_total_wrt_t(t,all_params["kobs"],all_params["concI0"],all_params["KI"],all_params["Etot"],uplim=all_params["uplim"])

    popt, pcov = curve_fit(fitting_adapter, t, occ_tot, p0=p0, bounds=tuple(bounds))
    fitted_data = fitting_adapter(t, *popt)
    fit_output = fitting._prepare_output(popt, pcov, param_order, t, fitted_data, occ_tot)

    return fit_output   

def kobs_wrt_concI0(concI0,KI,kinact,n=1): 
    '''
    Calculates the observed rate constant kobs with respect to the initial concentration of the inhibitor using a Michaelis-Menten-like equation.

    Args:
    concI0: Array of initial concentrations of the inhibitor.
    KI: Inhibition constant, analogous to K_M, K_D, and K_A, where kobs = kinact/2.
    kinact: Maximum potential rate of covalent bond formation.
    n (Optional): Hill coefficient, default is 1.

    Returns:
    np.array: Array of kobs values, the first order observed rate constants of inactivation, with units of inverse time.
    '''
    return kinact/(1+(KI/concI0)**n)

def KI_kinact_n_fit_to_kobs_wrt_concI0(concI0: np.ndarray, kobs: np.ndarray, nondefault_params: dict = None, xlim=None):
    """
    Fit parameters (KI, kinact, n) to kobs with respect to concI0 using a structured dictionary for parameters.

    Args:
    concI0: Array of initial concentrations of the inhibitor.
    kobs: Array of observed rate constants.
    nondefault_params (dict, optional): A structured dictionary of parameters with 'fixed','guess', and 'bound' keys. Include any params to update the default.
        default_params = {
            "KI": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
            "kinact": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
            "n": {"fix": 1, "guess": 1, "bounds": (-np.inf,np.inf)}, # fix overrides guess, so set fix to None if you wish to include this 
        }
    xlim (tuple): Limits for the concI0 points considered in the fit (min_concI0, max_concI0).
        
    Returns:
    an instance of the FitOutput class
    """
    # Default
    params = {
        "KI": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
        "kinact": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
        "n": {"fix": 1, "guess": 1, "bounds": (-np.inf,np.inf)}, 
    }

    if nondefault_params is not None:
        params = _update_dict_with_subset(params, nondefault_params)

    p0, bounds, param_order, fixed_params = fitting._extract_fit_info(params)

    if xlim:
        indices = (concI0 >= xlim[0]) & (concI0 <= xlim[1])
        concI0 = concI0[indices]
        kobs = kobs[indices]

    def fitting_adapter(concI0, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return kobs_wrt_concI0(concI0, all_params["KI"], all_params["kinact"], all_params["n"])

    popt, pcov = curve_fit(fitting_adapter, concI0, kobs, p0=p0, bounds=tuple(bounds))
    fitted_data = fitting_adapter(concI0, *popt)
    fit_output = fitting._prepare_output(popt, pcov, param_order, concI0, fitted_data, kobs)

    return fit_output

class Parameters:
    """
    Common place for parameters found in covalent inhibition literature.
    """
    @staticmethod
    def Ki(kon, koff):
        """
        Ki (i.e. inhib. dissociation constant, Kd) calculation
        
        Args:
        kon: on-rate constant (nM^-1*s^-1)
        koff: off-rate constant (s^-1)
        """
        return koff / kon

    @staticmethod
    def KI(kon, koff, kinact):
        """
        KI (i.e. inhibition constant, KM, KA, Khalf, KD) calculation.
        
        Args:
        kon: on-rate constant (nM^-1*s^-1)
        koff: off-rate constant (s^-1) 
        kinact: inactivation (last irrev step) rate constant
        """
        return (koff + kinact) / kon
    
    @staticmethod
    def CE(kon, koff, kinact):
        """
        Covalent efficiency (i.e. specificity, potency) calculation (kinact/KI)
        """
        return kinact/Parameters.KI(kon, koff, kinact)

class Experiments:
    """
    Common place for experimental setups in covalent inhibition literature.
    """
    #TODO: timecourse and KI/kinact 
    @staticmethod
    def timecourse(t,system):
        return NotImplementedError()