import numpy as np
from math import log10, floor
import sys
from io import StringIO
from itertools import product,islice


def rate_pair_from_P(int_OOM,P_B) -> tuple:
    """
    Provides rates between two states that are at rapid equilibrium and 
        therefore approximated by the population distribution. 

    Parameters:
    ----------
    int_OOM : int or float
        The order of magnitude of the rates.
    P_B : float
        The probability of the state B, i.e., kAB/(kBA+kAB) or [B]/([A]+[B]) for A<-->B.

    Returns:
    -------
    kAB : float
        The rate constant from A to B.
    kBA : float
        The rate constant from B to A.
    """
    kAB=P_B*10**(float(int_OOM)+1)
    kBA=10**(float(int_OOM)+1)-kAB
    return kAB,kBA

def integerable_float(num):
    """
    Returns a float as an integer if it is an integer, otherwise returns the float.
    """
    if num.is_integer():
        return int(num)
    else:
        return num
    
def round_sig(num: float, sig_figs: int = 3, autoformat=True):
    """
    Round up using significant figures.

    Parameters
    ----------
    num : float
        The number to be rounded.
    sig_figs : int
        The number of significant figures to round to.
    autoformat : bool, optional
        If True (default), formats the result into scientific notation if the order of magnitude 
        is greater than +/- 3.

    Returns
    -------
    float or str
        The rounded number. If autoformat is True and the result is in scientific notation, 
        it is returned as a string.
    """
    if num == 0:
        return 0.0

    order_of_magnitude = floor(log10(abs(num)))
    shift = sig_figs - 1 - order_of_magnitude

    # Scale the number, round it, and scale it back
    scaled_num = num * (10 ** shift)
    rounded_scaled_num = floor(scaled_num + 0.5)
    result = rounded_scaled_num / (10 ** shift)

    # Handling scientific notation conditionally
    if autoformat and (order_of_magnitude >= 4 or order_of_magnitude <= -4):
        return format(result, '.{0}e'.format(sig_figs - 1))
    else:
        return result

def update_dict(defaults: dict, updates: dict = None, subset=False) -> dict:
    """
    Recursively updates the default dictionary with values from the update dictionary.

    Parameters
    ----------
    defaults : dict
        The default dictionary containing initial key-value pairs.
    updates : dict
        The update dictionary containing key-value pairs to merge into the defaults dictionary.
    subset : bool
        If True, only updates keys present in defaults. If False, adds new keys from updates.

    Returns
    -------
    dict
        The merged dictionary.
    """
    if updates is None:
        return defaults

    for key, update_value in updates.items():
        if subset and key not in defaults:
            continue
        # Recurse through dictionaries
        if key in defaults and isinstance(defaults[key], dict) and isinstance(update_value, dict):
            defaults[key] = update_dict(defaults[key], update_value, subset=subset)
        else:
            defaults[key] = update_value
    return defaults
    

def compare_dictionaries(dict1, dict2, rel_tol=1e-9, abs_tol=0.0):
    """
    Compare two dictionaries recursively and check if their values are approximately equal.

    Parameters
    ----------
    dict1 : dict
        The first dictionary to compare.
    dict2 : dict
        The second dictionary to compare.
    rel_tol : float, optional
        The relative tolerance for comparing floating-point values. Default is 1e-9.
    abs_tol : float, optional
        The absolute tolerance for comparing floating-point values. Default is 0.0.

    Returns
    -------
    bool
        True if the dictionaries are approximately equal, False otherwise.
    """
    
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, dict) and isinstance(value2, dict):
            # Recursive call for nested dictionaries
            if not compare_dictionaries(value1, value2, rel_tol, abs_tol):
                return False
        else:
            arr1, arr2 = np.array(value1), np.array(value2)
            if not np.allclose(arr1, arr2, rtol=rel_tol, atol=abs_tol):
                return False
    return True



def generate_dictval_combinations(input_dict: dict):
    """
    Generate all combinations of values in a dictionary and return them as a dictionary
    with the original keys and numpy arrays of corresponding values.

    Parameters
    ----------
    input_dict : dict
        Keys are parameter names and values are numpy arrays.

    Returns
    -------
    dict
        Dictionary with the original keys and numpy arrays of corresponding value
    """

    names = list(input_dict.keys())
    values = list(input_dict.values())

    combinations = list(product(*values))
    combinations_array = np.array(combinations).T

    comb_dict = {name: combinations_array[i] for i, name in enumerate(names)}
    return comb_dict

def arr2float(value: np.ndarray):
    """
    Convert an size-1 array to float.
    """
    if isinstance(value,np.ndarray):
        if value.size == 1:
            return float(value[0])
        else:
            raise ValueError("Array is not size 1.")
    else:
        return value


def chunks(data, size):
    """
    Split data into chunks with a specified size.
    """
    it = iter(data)
    for _ in range(0, len(data), size):
        yield list(islice(it, size))

def printer(queue):
    """Function to print messages from the queue."""
    while True:
        msg = queue.get()
        if msg == "DONE":
            break
        print(msg)

class CaptureOutput(list):
    """Custom output stream to capture prints."""
    def __enter__(self):
        self._stdout = sys.stdout
        self._stringio = StringIO()
        sys.stdout = self._stringio
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout