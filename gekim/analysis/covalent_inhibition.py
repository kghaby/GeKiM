import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from ..utils.helpers import update_dict_with_subset
from .fitting import detect_bad_fit, FitOutput, _normalize_params, _unnormalize_popt, _extract_fit_info, _prepare_output

#TODO: fit to scheme. meaning yuo make a scheme without values for the transitions and fit it to occ data to see what values of rates satisfy curve
#TODO: detect trivial solutions for curve fitting, like if all values are the same, or if all values are 0, or if all values are 1.
#TODO: time arrays that are not evenly spaced will hurt curve fitting.
#TODO: refactor fitting. kobs fitting shares lots

def occ_final_wrt_t(t,kobs,Etot,uplim=1):
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
    return uplim*Etot*(1-np.e**(-kobs*t))

def kobs_uplim_fit_to_occ_final_wrt_t(t: np.ndarray, occ_final: np.ndarray, nondefault_params: dict = None, xlim: tuple = None, 
                                      normalize_for_fit=True, sigma_kde=False, sigma=None, **kwargs) -> FitOutput: 
    '''
    Fit kobs to the first order occupancy over time.

    Parameters
    ----------
    t : np.ndarray
        Array of timepoints.
    occ_final : np.ndarray
        Array of observed occupancy, i.e. concentration.
    nondefault_params : dict, optional
        A structured dictionary of parameters with 'fixed','guess', and 'bound' keys. 
        Include any params to override the default.
        ```python
        default_params = {
            # Observed rate constant
            "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)}, 
            # Total concentration of E over all species
            "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},    
            # Scales the upper limit of the curve
            "uplim": {"fix": None, "guess": 1, "bounds": (0,np.inf)},   
        }
        ```
    xlim : tuple, optional
        Limits for the time points considered in the fit (min_t, max_t).
    normalize_for_fit : bool, optional
        If True, normalize the observed data and relevant params by dividing by the maximum value before fitting. 
        Will still return unnormalized values. Default is True.
    sigma_kde : bool, optional
        If True, calculate the density of the x-values and use that for sigma (uncertainty) in the curve_fitting. 
        Helps distribute weight over unevenly-spaced points. Default is False.
    sigma : np.ndarray, optional
        sigma parameter for curve_fit. This argument is overridden if sigma_kde=True. Default is None.
    kwargs : dict, optional
        Additional keyword arguments to pass to the curve_fit function.

    Returns
    -------
    FitOutput
        An instance of the FitOutput class

    Example
    -------
    ```python
    fit_output =  ci.kobs_uplim_fit_to_occ_final_wrt_t(t,
                    system.system.species["EI"].simout["y"],
                    nondefault_params={"Etot":{"fix":concE0}})
    ```
    will fit kobs and uplim to the concentration of EI over time, fixing Etot at concE0.

    Notes
    -----
    "fix" takes priority over "guess" in the param dict.
    '''

    # Default
    params = {
        "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
        "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},
        "uplim": {"fix": None, "guess": 1, "bounds": (0,np.inf)}, 
    }

    if nondefault_params is not None:
        params = update_dict_with_subset(params, nondefault_params)


    if normalize_for_fit:
        norm_factor = occ_final.max()
        occ_final_unnorm = occ_final
        occ_final = occ_final/norm_factor
        params = _normalize_params(params,norm_factor,["Etot"])

    p0, bounds, param_order, fixed_params = _extract_fit_info(params)

    if xlim:
        indices = (t >= xlim[0]) & (t <= xlim[1])
        t = t[indices]
        occ_final = occ_final[indices]
    
    if sigma_kde:
        kde = gaussian_kde(t)
        sigma = kde(t) 
    else:
        sigma = sigma

    def fitting_adapter(t, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return occ_final_wrt_t(t,all_params["kobs"],all_params["Etot"],uplim=all_params["uplim"])
    
    #jac_func = fitting.generate_jacobian_func(fitting_adapter, param_order) #makes curve_fit almost 2x slower 
    popt, pcov = curve_fit(fitting_adapter, t, occ_final, p0=p0, bounds=bounds, sigma=sigma, **kwargs)

    # Unnorm occ_final and params
    if normalize_for_fit:
        occ_final = occ_final_unnorm
        for param in ["Etot"]:
            if param in param_order:
                popt = _unnormalize_popt(popt,param_order,norm_factor,[param])
            elif param in fixed_params:
                fixed_params[param] = fixed_params[param]*norm_factor
    
    fitted_data = fitting_adapter(t, *popt)
    fit_output = _prepare_output(t, fitted_data, occ_final, popt, pcov, param_order)

    bad_fit, message = detect_bad_fit(fitted_data, occ_final, popt, pcov, bounds, param_order)
    if bad_fit:
        print(f"Bad fit detected:{message}")
        print(f"\tFitted params: {fit_output.fitted_params}\n")

    return fit_output



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

    FO = 1/(1+(KI/concI0)) # Equilibrium occupancy of reversible portion
    return uplim*Etot*(1-(1-FO)*(np.e**(-kobs*t)))

def kobs_KI_uplim_fit_to_occ_total_wrt_t(t: np.ndarray, occ_tot: np.ndarray, nondefault_params: dict = None, xlim: tuple = None, 
                                         normalize_for_fit=True, sigma_kde=False, sigma=None, **kwargs) -> FitOutput: 
    '''
    Fit kobs and KI to the total occupancy of all bound states over time, 
    assuming fast reversible binding equilibrated at t=0.

    Parameters
    ----------
    t : np.ndarray
        Array of timepoints.
    occ_tot : np.ndarray
        Array of total occupancy.
    nondefault_params : dict, optional
        A structured dictionary of parameters with 'fixed','guess', and 'bound' keys. 
        Include any params to override the default.
        ```python
        default_params = {
            # Observed rate constant
            "kobs": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
            # Initial concentration of the (saturating) inhibitor
            "concI0": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
            # Inhibition constant where kobs = kinact/2.
            "KI": {"fix": None, "guess": 10, "bounds": (0,np.inf)},
            # Total concentration of E across all species
            "Etot": {"fix": None, "guess": 1, "bounds": (0,np.inf)},
            # Scales the upper limit of the curve
            "uplim": {"fix": None, "guess": 1, "bounds": (0,1)},        
        }
        ```
    xlim : tuple, optional
        Limits for the time points considered in the fit (min_t, max_t).
    normalize_for_fit : bool, optional
        If True, normalize the observed data and relevant params by dividing by the maximum value before fitting. 
        Will still return unnormalized values. Default is True.
    sigma_kde : bool, optional
        If True, calculate the density of the x-values and use that for sigma (uncertainty) in the curve_fitting. 
        Helps distribute weight over unevenly-spaced points. Default is False.
    sigma : np.ndarray, optional
        sigma parameter for curve_fit. This argument is overridden if sigma_kde=True. Default is None.
    kwargs : dict, optional
        Additional keyword arguments to pass to the curve_fit function.

    Returns
    -------
    FitOutput
        An instance of the FitOutput class
    
    Notes
    -----
    "fix" takes priority over "guess" in the param dict.
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
        params = update_dict_with_subset(params, nondefault_params)

    if normalize_for_fit:
        norm_factor = occ_tot.max()
        occ_tot_unnorm = occ_tot
        occ_tot = occ_tot/norm_factor
        params = _normalize_params(params,norm_factor,["Etot"])
    
    p0, bounds, param_order, fixed_params = _extract_fit_info(params)

    if xlim:
        indices = (t >= xlim[0]) & (t <= xlim[1])
        t = t[indices]
        occ_tot = occ_tot[indices]

    if sigma_kde:
        kde = gaussian_kde(t)
        sigma = kde(t) 
    else:
        sigma = sigma

        
    def fitting_adapter(t, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return occ_total_wrt_t(t,all_params["kobs"],all_params["concI0"],all_params["KI"],all_params["Etot"],uplim=all_params["uplim"])

    popt, pcov = curve_fit(fitting_adapter, t, occ_tot, p0=p0, bounds=bounds, sigma=sigma, **kwargs)

    # Unnorm occ and params
    if normalize_for_fit:
        occ_tot = occ_tot_unnorm
        for param in ["Etot"]:
            if param in param_order:
                popt = _unnormalize_popt(popt,param_order,norm_factor,[param])
            elif param in fixed_params:
                fixed_params[param] = fixed_params[param]*norm_factor
    
    fitted_data = fitting_adapter(t, *popt)
    fit_output = _prepare_output(t, fitted_data, occ_tot, popt, pcov, param_order,)
    
    bad_fit, message = detect_bad_fit(fitted_data, occ_tot, popt, pcov, bounds, param_order)
    if bad_fit:
        print(f"Bad fit detected:{message}")
        print(f"\tFitted params: {fit_output.fitted_params}\n")
    
    return fit_output   

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
    n : int, optional
        Hill coefficient, default is 1.

    Returns
    -------
    np.ndarray
        Array of kobs values, the first order observed rate constants of inactivation, 
        with units of inverse time.
    '''
    return kinact/(1+(KI/concI0)**n)

def KI_kinact_n_fit_to_kobs_wrt_concI0(concI0: np.ndarray, kobs: np.ndarray, nondefault_params: dict = None, xlim: tuple = None,
                                       normalize_for_fit=True, sigma_kde=False, sigma=None, **kwargs) -> FitOutput: 
    """
    Fit parameters (KI, kinact, n) to kobs with respect to concI0 using 
    a structured dictionary for parameters.

    Parameters
    ----------
    concI0 : np.ndarray
        Array of initial concentrations of the inhibitor.
    kobs : np.ndarray
        Array of observed rate constants.
    nondefault_params : dict, optional
        A structured dictionary of parameters with 'fixed','guess', and 'bound' keys. 
        Include any params to override the default.
        ```python
        default_params = {
            "KI": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
            "kinact": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
            # fix overrides guess, so set fix to None if you wish to include "n"
            "n": {"fix": 1, "guess": 1, "bounds": (-np.inf,np.inf)}, 
        }
        ```
    xlim : tuple, optional
        Limits for the concI0 points considered in the fit (min_concI0, max_concI0).
    normalize_for_fit : bool, optional
        If True, normalize the observed data and relevant params by dividing by the maximum value before fitting. 
        Will still return unnormalized values. Default is True.
    sigma_kde : bool, optional
        If True, calculate the density of the x-values and use that for sigma (uncertainty) in the curve_fitting. 
        Helps distribute weight over unevenly-spaced points. Default is False.
    sigma : np.ndarray, optional
        sigma parameter for curve_fit. This argument is overridden if sigma_kde=True. Default is None.
    kwargs : dict, optional
        Additional keyword arguments to pass to the curve_fit function.    
    
    Returns
    -------
    FitOutput
        An instance of the FitOutput class
    
    Notes
    -----
    "fix" takes priority over "guess" in the param dict.
    """
    # Default
    params = {
        "KI": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
        "kinact": {"fix": None, "guess": 0.01, "bounds": (0,np.inf)},
        "n": {"fix": 1, "guess": 1, "bounds": (-np.inf,np.inf)}, 
    }

    if nondefault_params is not None:
        params = update_dict_with_subset(params, nondefault_params)

    if normalize_for_fit:
        norm_factor = kobs.max()
        kobs_unnorm = kobs
        kobs = kobs/norm_factor
        params = _normalize_params(params,norm_factor,["kinact"])

    p0, bounds, param_order, fixed_params = _extract_fit_info(params)

    if xlim:
        indices = (concI0 >= xlim[0]) & (concI0 <= xlim[1])
        concI0 = concI0[indices]
        kobs = kobs[indices]

    if sigma_kde:
        kde = gaussian_kde(concI0)
        sigma = kde(concI0) 
    else:
        sigma = sigma

    def fitting_adapter(concI0, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return kobs_wrt_concI0(concI0, all_params["KI"], all_params["kinact"], all_params["n"])

    popt, pcov = curve_fit(fitting_adapter, concI0, kobs, p0=p0, bounds=bounds, sigma=sigma, **kwargs)
        
    # Unnorm kobs and params
    if normalize_for_fit:
        kobs = kobs_unnorm
        for param in ["kinact"]:
            if param in param_order:
                popt = _unnormalize_popt(popt,param_order,norm_factor,[param])
            elif param in fixed_params:
                fixed_params[param] = fixed_params[param]*norm_factor
            
    fitted_data = fitting_adapter(concI0, *popt)
    fit_output = _prepare_output(concI0, fitted_data, kobs, popt, pcov, param_order)
    
    bad_fit, message = detect_bad_fit(fitted_data, kobs, popt, pcov, bounds, param_order)
    if bad_fit:
        print(f"Bad fit detected:{message}")
        print(f"\tFitted params: {fit_output.fitted_params}\n")
    
    return fit_output

class Parameters:
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
        return kinact/Parameters.KI(kon, koff, kinact)

class Experiments:
    """
    Common place for experimental setups in covalent inhibition literature.
    """
    #TODO: timecourse and KI/kinact 
    @staticmethod
    def timecourse(t,system):
        return NotImplementedError()