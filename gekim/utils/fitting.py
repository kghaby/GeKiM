import numpy as np
from inspect import signature
from sympy import symbols, Matrix, lambdify
from typing import Union, Callable
from scipy.stats import gaussian_kde
from lmfit import Model, Parameters
from lmfit.model import ModelResult

from .helpers import arr2float


#TODO: scheme fitting, rate constant fitting
                
def general_fit(model_func: Callable, x: np.ndarray, y: np.ndarray, 
                params: Union[dict, Parameters], xlim: tuple = None, 
                weights_kde=False, weights: np.ndarray = None,
                verbosity=2, independent_vars: list[str] = None, model_kwargs: dict = None, **kwargs) -> ModelResult:
    """
    General fitting function using lmfit.

    Parameters
    ----------
    model_func : callable
        The model function to fit. Should be of the form model_func(indep_var, *args, **kwargs).
    x : np.ndarray
        Array of independent variable.
    y : np.ndarray
        Array of observed data.
    params : dict
        Dictionary of parameters with 'fixed', 'guess', and 'bounds' keys.
    xlim : tuple, optional
        Limits for the x points considered in the fit (min_x, max_x).
    weights_kde : bool, optional
        If True, calculate the density of the x-values and use the normalized reciprocol as weights. Similar to 1/sigma for scipy.curve_fit.
        Helps distribute weight over unevenly-spaced points. Default is False.
    weights : np.ndarray, optional
        weights parameter for fitting. This argument is overridden if weights_kde=True. Default is None.
    verbosity : int, optional
        0: print nothing. 1: print upon bad fit. 2: print always. Default is 2.
    independent_vars : list of str, optional
        List of independent variable names to pass to the lmfit Model. Default is None, and will use 
        the first parameter name of the model function.
    model_kwargs : dict, optional
        Keyword arguments used in the model function that are not parameters. Default is None. 
        Will be passed to the lmfit Model initialization.
    kwargs : dict, optional
        Additional keyword arguments to pass to the lmfit Model.fit function.

    Returns
    -------
    ModelResult
        The result of the fitting operation from lmfit.

    Notes
    -----
    The x-data used in the fit is stored in the ModelResult object as result.userdata['x'] for consistency.
    It also exists as result.userkws[first_indep_var_name].
    """ 
    if xlim:
        indices = (x >= xlim[0]) & (x <= xlim[1])
        x = x[indices]
        y = y[indices]

    if weights_kde:
        kde = gaussian_kde(x)
        weights = 1/kde(x)
        weights /= np.sum(weights)  # Normalize

    else:
        weights = weights

    if isinstance(params, Parameters):
        lm_params = params
    elif isinstance(params, dict):
        lm_params = Parameters()
        for name, info in params.items():
            if "vary" in info and not info["vary"]:
                lm_params.add(name, value=info["value"], vary=False)
            else:
                lm_params.add(name, value=info["value"], min=info["bounds"][0], max=info["bounds"][1])
    else:
        raise ValueError("params must be a lmfit.Parameters or dict instance.")

    # Set empty model_kwargs to an empty dict bc default values should not be mutable
    if model_kwargs is None:
        model_kwargs = {} 

    # Dynamically handle the independent variable name so that model funcs aren't required to use "x"
    sig = signature(model_func)
    first_indep_var_name = list(sig.parameters.keys())[0]
    if independent_vars is None:
        independent_vars = [first_indep_var_name]
    model = Model(model_func,independent_vars=independent_vars, param_names=list(lm_params.keys()),**model_kwargs)
    model_result = model.fit(y, lm_params, **{first_indep_var_name: x}, weights=weights, **kwargs)

    # Future proof storing x data in the ModelResult object
    if hasattr(model_result, "userdata"):
        print("Warning: model_result.userdata already exists.")
        if "x" in model_result.userdata:
            print("Warning: model_result.userdata['x'] already exists. This is unexpected.")
            print(f"\tNot overwriting since x-data is also stored in model_result.userkws['{first_indep_var_name}']"
                    f"(or under any other independent variable name).")
        else:
            model_result.userdata["x"] = x
    else:
        model_result.userdata = {"x": x}

    if verbosity >= 1:
        bad_fit, message = detect_bad_fit(model_result)
        if bad_fit:
            print(f"Bad fit detected: {message}\n")

        if verbosity >= 2 or bad_fit:
            model_result.params.pretty_print();print('\n')


    return model_result

def merge_params(default_params: Parameters, nondefault_params: Union[dict, Parameters] = None) -> Parameters:
    """
    Merge default and nondefault parameters into a Parameters object.

    Parameters
    ----------
    default_params : Parameters
        The default parameters.
    nondefault_params : dict or Parameters, optional
        The nondefault parameters to override the defaults.

    Returns
    -------
    Parameters
        Merged Parameters object.
    """
    if nondefault_params is None:
        return default_params

    merged_params = default_params.copy()


    if isinstance(nondefault_params, Parameters):
        for name, param in nondefault_params.items():
            if name in merged_params:
                merged_params[name].set(
                    value=arr2float(param.value),
                    vary=param.vary,
                    min=param.min,
                    max=param.max,
                    expr=param.expr,
                    brute_step=param.brute_step,
                )
            else:
                merged_params.add(name,
                    value=arr2float(param.value),
                    vary=param.vary,
                    min=param.min,
                    max=param.max,
                    expr=param.expr,
                    brute_step=param.brute_step,
                )
    elif isinstance(nondefault_params, dict):
        for name, info in nondefault_params.items():
            if name in merged_params:
                merged_params[name].set(
                    value=arr2float(info.get("value", merged_params[name].value)),
                    vary=info.get("vary", merged_params[name].vary),
                    min=info.get("min", merged_params[name].min),
                    max=info.get("max", merged_params[name].max),
                    expr=info.get("expr", merged_params[name].expr),
                    brute_step=info.get("brute_step", merged_params[name].brute_step),
                )
            else:
                merged_params.add(name,
                    value=arr2float(info.get("value")),
                    vary=info.get("vary", True),
                    min=info.get("min", -np.inf),
                    max=info.get("max", np.inf),
                    expr=info.get("expr", None),
                    brute_step=info.get("brute_step", None),
                )
    else:
        raise ValueError("nondefault_params must be a dict or lmfit.Parameters object.")

    return merged_params

def chi_squared(observed_data: np.ndarray, fitted_data: np.ndarray, fitted_params: np.ndarray, variances: np.ndarray = None, reduced=False):
    """
    Calculate the chi-squared and optionally the reduced chi-squared statistics.

    Parameters
    ----------
    observed_data : np.ndarray
        The observed data points.
    fitted_data : np.ndarray
        The fitted data points obtained from curve 
    fitted_params : list or np.ndarray
        The optimal parameters obtained from curve 
    variances : np.ndarray, optional
        Variances of the observed data points. If None, assume constant variance.
    reduced : bool, optional
        If True, calculate and return the reduced chi-squared.

    Returns
    -------
    float
        The chi-squared or reduced chi-squared statistic.
    """

    if len(observed_data) != len(fitted_data):
        raise ValueError("Length of observed_data and fitted_data must be the same.")

    if len(fitted_params) == 0:
        raise ValueError("fitted_params cannot be empty.")

    residuals = observed_data - fitted_data
    chi_squared = np.sum((residuals**2) / variances) if variances is not None else np.sum(residuals**2)

    if reduced:
        degrees_of_freedom = len(observed_data) - len(fitted_params)
        if degrees_of_freedom <= 0:
            raise ValueError("Degrees of freedom must be positive.")
        return chi_squared / degrees_of_freedom

    return chi_squared

def _extract_fit_info(params):
    """Extracts initial guesses and bounds, and separates fixed and fitted parameters."""
    p0, bounds, param_order = [], ([], []), []
    fixed_params = {}
    for param, config in params.items():
        if config["fix"] is not None:
            fixed_params[param] = config["fix"]
        else:
            p0.append(config["guess"])
            bounds[0].append(config["bounds"][0])
            bounds[1].append(config["bounds"][1])
            param_order.append(param)
    return p0, bounds, param_order, fixed_params

class FitOutput:
    """
    Comprises the output of a fitting operation.

    Attributes
    ----------
    fitted_params : dict
        Parameters obtained from the fit, keyed by parameter name and zipped in the order indexed in pcov.
    pcov : array
        Covariance matrix of the fitted parameters.
    x : array
        Independent variable used for fitting.
    y_fit : array
        Data generated by the fitted params.
    y_obs : array
        Observed data used for fitting. May be normalized.
    reduced_chi_sq : float
        Reduced chi-squared value indicating goodness of fit.
    """
    def __init__(self, x, fitted_ydata, observed_ydata, fitted_params, pcov, reduced_chi_sq):
        self.fitted_params = fitted_params
        self.pcov = pcov
        self.x = x
        self.y_fit = fitted_ydata
        self.y_obs = observed_ydata
        self.reduced_chi_sq = reduced_chi_sq

def _prepare_output(x, fitted_ydata, observed_ydata, popt, pcov, param_order):
    """
    Prepare the fitting output as an instance of the FitOutput class.
    """
    fitted_params = dict(zip(param_order, popt))
    reduced_chi_sq = chi_squared(observed_ydata, fitted_ydata, popt, np.var(observed_ydata), reduced=True)
    return FitOutput(x, fitted_ydata, observed_ydata, fitted_params, pcov, reduced_chi_sq)

def generate_jacobian_func(fitting_adapter, param_order):
    '''
    Generate and store the Jacobian function for a given model function.
    Makes curve_fit slower for simple functions.

    Parameters
    ----------
    fitting_adapter : callable
        The fitting adapter function to fit. Should be of the form fitting_adapter(x, *params).
    param_order : list of str
        List of parameter names for the fitting adapter function.

    Returns
    -------
    jac_func : Callable
        A function that calculates the Jacobian matrix given the parameters.
    '''
    # Inspect the model function to get parameter names
    sig = signature(fitting_adapter)
    x_sym = list(sig.parameters.keys())[0]          # Independent variable
    x_sym = symbols(x_sym)
    param_syms = symbols(param_order)
    
    model_expr = fitting_adapter(x_sym, *param_syms)
    
    # Compute the Jacobian 
    jacobian_matrix = Matrix([model_expr]).jacobian(param_syms)
    jacobian_func = lambdify((x_sym, *param_syms), jacobian_matrix, 'numpy')
    
    # Wrapper to match curve_fit expected format
    def jac_func(x, *params):
        jacobian_values = [jacobian_func(xi, *params) for xi in x]
        return np.array(jacobian_values, dtype=float).squeeze()

    return jac_func

def _normalize_params(params: dict, norm_factor: float,names: list) -> dict:
    """Normalize the parameter values and bounds by the maximum value of the observable."""
    for name in names:
        if params[name]["fix"] is not None:
            params[name]["fix"] = params[name]["fix"]/norm_factor
        else:
            params[name]["guess"] = params[name]["guess"]/norm_factor
            params[name]["bounds"] = (params[name]["bounds"][0]/norm_factor,params[name]["bounds"][1]/norm_factor)
    return params

def _unnormalize_popt(popt,param_order,norm_factor,names):
    for name in names:
        index = param_order.index(name)
        popt[index] = popt[index]*norm_factor
    return popt

def calc_weights_dt(t: np.ndarray):
    """
    Calculate weights as the inverse of the differences in time points

    Parameters
    ----------
    t : np.ndarray
        Array of time points.

    Returns
    -------
    weights : np.ndarray
        Array of weights. May be used as sigma in fitting functions.
    """
    # 
    dt = np.diff(t, prepend=t[1]-t[0])
    weights = 1 / dt
    weights /= np.sum(weights)  # Normalize 
    return weights

def detect_bad_fit(result: ModelResult, atol: float = 1e-10) -> tuple[bool, str]:
    """
    Detects issues with the fit result from lmfit.

    Parameters
    ----------
    result : ModelResult
        The result of the fitting operation from lmfit.

    Returns
    -------
    bad : bool
    message : str
    """
    bad = False
    message = ""

    # Extract fitted data and observed data
    fitted_data = result.best_fit
    y_obs = result.data
    residuals = result.residual

    # Check for constant output
    if np.allclose(fitted_data, fitted_data[0], atol=atol):
        bad = True
        message += f"\n\tModel fit y-values were all within {atol:.2e} of each other."

    # Residual analysis
    if np.any(np.abs(residuals) > 10 * np.std(y_obs)):
        bad = True
        message += "\n\tResiduals are too large."

    # Check if parameters are at bounds
    for name, param in result.params.items():
        if not param.vary:
            continue
        if np.isclose(param.value, param.min):
            bad = True
            message += f"\n\tParameter {name} is at or near its lower bound: {param.min}."
        elif np.isclose(param.value, param.max):
            bad = True
            message += f"\n\tParameter {name} is at or near its upper bound: {param.max}."

    # Check standard errors
    if result.errorbars:
        param_errors = {name: param.stderr for name, param in result.params.items() if param.stderr is not None}
        for name, param in result.params.items():
            if name in param_errors and param_errors[name] > abs(param.value):
                bad = True
                message += f"\n\tLarge standard errors relative to parameter values:"
                message += f"\n\t\t{name}: {param.value:.2e} ± {param_errors[name]:.2e}"

    # Check R^2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r_squared = 1 - (ss_res / ss_tot)
    if r_squared < 0.5:
        bad = True
        message += f"\n\tLow R² value: {r_squared:.2e}"

    return bad, message

def calc_nrmse(y_exp: np.ndarray, y_pred: np.ndarray):
    """
    Calculate the Normalized Root Mean Squared Error (NRMSE) between two arrays.

    Parameters
    ----------
    y_exp : np.ndarray
        Array of experimental values.
    y_pred : np.ndarray
        Array of predicted values.

    Returns
    -------
    float
        NRMSE value between 0 and 1, where 1 is a perfect match.
    """
    mse = np.mean((y_exp - y_pred) ** 2)
    rmse = np.sqrt(mse)
    range_y = np.ptp(y_exp)  # Equivalent to max(y_exp) - min(y_exp)
    nrmse = 1 - rmse / range_y
    return nrmse

