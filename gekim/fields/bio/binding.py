import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from ...utils.helpers import update_dict_with_subset
from ...utils.fitting import detect_bad_fit, FitOutput, _normalize_params, _unnormalize_popt, _extract_fit_info, _prepare_output

#TODO: normalize for fit for x axis as well

def res_time():
    raise NotImplementedError()

def dose_response(dose: np.ndarray,Khalf: float,n=1,uplim=1): 
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

    Returns
    -------
    np.ndarray
        The fraction of the responding population.
    '''
    return uplim/(1+(Khalf/dose)**n)

def dose_response_fit(dose: np.ndarray, response: np.ndarray, nondefault_params: dict = None, xlim: tuple = None,
                                       normalize_for_fit=True, sigma_kde=False, sigma=None, **kwargs) -> FitOutput: 
    """
    Fit parameters (Khalf, Khalfnact, n) to response with respect to dose using 
    a structured dictionary for parameters.

    Parameters
    ----------
    dose : np.ndarray
        Array of input concentrations of the ligand.
    response : np.ndarray
        Array of the fraction of the responding population.
    nondefault_params : dict, optional
        A structured dictionary of parameters with 'fixed','guess', and 'bound' keys. 
        Include any params to override the default.
        ```python
        default_params = {
            "Khalf": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
            "n": {"fix": None, "guess": 1, "bounds": (0,np.inf)}, 
            "uplim": {"fix": 1, "guess": 1, "bounds": (0,np.inf)},
        }
        ```
    xlim : tuple, optional
        Limits for the dose points considered in the fit (min_dose, max_dose).
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
        "Khalf": {"fix": None, "guess": 100, "bounds": (0,np.inf)},
        "n": {"fix": None, "guess": 1, "bounds": (0,np.inf)}, 
        "uplim": {"fix": 1, "guess": 1, "bounds": (0,np.inf)},
    }

    if nondefault_params is not None:
        params = update_dict_with_subset(params, nondefault_params)

    if normalize_for_fit:
        norm_factor = response.max()
        response_unnorm = response
        response = response/norm_factor
        params = _normalize_params(params,norm_factor,["uplim"]) # any params that need to be normalized

    p0, bounds, param_order, fixed_params = _extract_fit_info(params)

    if xlim:
        indices = (dose >= xlim[0]) & (dose <= xlim[1])
        dose = dose[indices]
        response = response[indices]

    if sigma_kde:
        kde = gaussian_kde(dose)
        sigma = kde(dose) 
    else:
        sigma = sigma

    def fitting_adapter(dose, *fitting_params):
        all_params = {**fixed_params, **dict(zip(param_order, fitting_params))}
        return dose_response(dose, all_params["Khalf"], all_params["n"], all_params["uplim"])

    popt, pcov = curve_fit(fitting_adapter, dose, response, p0=p0, bounds=bounds, sigma=sigma, **kwargs)
        
    # Unnorm response and params
    if normalize_for_fit:
        response = response_unnorm
        for param in ["uplim"]: # any params that need to be unnormalized 
            if param in param_order:
                popt = _unnormalize_popt(popt,param_order,norm_factor,[param])
            elif param in fixed_params:
                fixed_params[param] = fixed_params[param]*norm_factor
            
    fitted_data = fitting_adapter(dose, *popt)
    fit_output = _prepare_output(dose, fitted_data, response, popt, pcov, param_order)
    
    bad_fit, message = detect_bad_fit(fitted_data, response, popt, pcov, bounds, param_order)
    if bad_fit:
        print(f"Bad fit detected:{message}")
        print(f"\tFitted params: {fit_output.fitted_params}\n")
    
    return fit_output

class Parameters:
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

    
