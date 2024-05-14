import numpy as np
import math

#TODO: pathfinder that counts and lists all possible paths between two species

def rate_pair_from_P(intOOM,Pf):
    """
    Provides rates between two states that are at rapid equilibrium and therefore approximated by the population distribution. 
    intOOM is the order of magnitude of the rates.
    Pf is the probability of the forward state, ie kf/(kf+kb) or [B]/([A]+[B]) for A<-->B.
    """
    kf=Pf*10**(float(intOOM)+1)
    kb=10**(float(intOOM)+1)-kf
    return kf,kb

def integerable_float(num):
    """
    Returns a float as an integer if it is an integer, otherwise returns the float.
    """
    if num.is_integer():
        return int(num)
    else:
        return num
    
def round_sig(num, sig_figs, autoformat=True):
    """
    Round up using significant figures.
    autoformat = True (default) formats into scientific notation if the order of magnitude more than +/- 3.
    """
    if num == 0:
        return 0.0

    order_of_magnitude = math.floor(math.log10(abs(num)))
    shift = sig_figs - 1 - order_of_magnitude

    # Scale the number, round it, and scale it back
    scaled_num = num * (10 ** shift)
    rounded_scaled_num = math.floor(scaled_num + 0.5)
    result = rounded_scaled_num / (10 ** shift)

    # Handling scientific notation conditionally
    if autoformat and (order_of_magnitude >= 4 or order_of_magnitude <= -4):
        return format(result, '.{0}e'.format(sig_figs - 1))
    else:
        return result

def _update_dict_with_subset(defaults: dict, updates: dict):
    """
    Recursively updates the default dictionary with values from the update dictionary,
    ensuring that only the keys present in the defaults are updated.

    Args:
    defaults: The default dictionary containing all allowed keys with their default values.
    updates: The update dictionary containing keys to update in the defaults dictionary.

    Returns:
    dict: The updated dictionary.
    """

    for key, update_value in updates.items():
        if key in defaults:
            # If both the default and update values are dictionaries, recurse
            if isinstance(defaults[key], dict) and isinstance(update_value, dict):
                defaults[key] = _update_dict_with_subset(defaults[key], update_value)
            else:
                defaults[key] = update_value

    return defaults

def compare_dictionaries(dict1, dict2, rel_tol=1e-9, abs_tol=0.0):
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

