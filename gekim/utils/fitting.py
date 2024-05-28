import numpy as np
import inspect
from sympy import symbols, Matrix, lambdify
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from .helpers import update_dict_with_subset

#TODO: scheme fitting, rate constant fitting

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
    sig = inspect.signature(fitting_adapter)
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

def detect_bad_fit(fitted_data, y_obs, popt, pcov, param_bounds, param_order):
    bad = False
    message = ""

    # Check for constant output
    atol = 1e-10
    if np.allclose(fitted_data, fitted_data[0],atol=atol):
        bad = True
        message += f"\n\tModel fit y-values were all within {atol:.2e} of eachother."

    # Residual analysis
    residuals = y_obs - fitted_data
    if np.any(np.abs(residuals) > 10 * np.std(y_obs)):
        bad = True
        message += "\n\tResiduals are too large."

    # Check if parameters are at bounds
    for i, param in enumerate(popt):
        lower_bound, upper_bound = param_bounds[0][i], param_bounds[1][i]
        if np.isclose(param, lower_bound) or np.isclose(param, upper_bound):
            bad = True
            message += f"\n\tParameter {param_order[i]} is at or near its bound."

    # Check standard errors
    param_errors = np.sqrt(np.diag(pcov))
    if np.any(param_errors > np.abs(popt)):
        bad = True
        message += "\n\tLarge standard errors relative to parameter values (note that this could be due to absolute_sigma=True if you used this kwarg for curve_fit):"
        for i,param in enumerate(param_order):
            if param_errors[i] > np.abs(popt[i]):
                message += f"\n\t\t{param}: {popt[i]:.2e} ± {param_errors[i]:.2e}"

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