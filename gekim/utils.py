import numpy as np
import math
import colorsys

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

class Plotting:
    def assign_colors_to_species(schemes: dict, saturation_range: tuple = (0.5, 0.7), lightness_range: tuple = (0.3, 0.4), 
                                method: str = None, offset: float = 0, overwrite_existing=False, seed: int = None):
        """
        Assigns a distinct and aesthetically pleasing color to each species in a dictionary or a single kinetic scheme.
        Uses either a fixed or golden ratio based distribution for hues. Optionally seeds the randomness for consistent results.

        Args:
        schemes (dict): Dictionary of kinetic scheme dictionaries or a single kinetic scheme dictionary.
        saturation_range (tuple): Min and max saturation values.
        lightness_range (tuple): Min and max lightness values.
        method (str): "GR" for golden ratio hue distribution; None for linear distribution.
        offset (float): Offset value for the hues.
        overwrite_existing (bool): If True, overwrite existing colors; if False, assign colors only to species without colors.
        seed (int): Seed for random number generator for reproducible color variations.

        Returns:
        dict: Updated schemes with assigned colors. Edits original input dict.
        """
        #TODO: handle list of schemes and NState class
        if not isinstance(schemes, dict):
            raise ValueError("Input should be a dictionary of schemes or a single scheme formatted as a dictionary.")
        
        single_scheme = False
        if 'species' in next(iter(schemes.keys())):
            single_scheme = True
            schemes = {'single_scheme': schemes}

        # Retrieve unique species into a list. Don't use a set to preserve order.
        unique_species = []
        seen_species = set()
        for scheme in schemes.values():
            for species in scheme["species"].keys():
                if species not in seen_species:
                    unique_species.append(species)
                    seen_species.add(species)

        golden_ratio_conjugate = 0.618033988749895
        hue = 0

        n_colors = len(unique_species)
        color_mapping = {}
        hues = np.linspace(0, 1, n_colors, endpoint=False)

        # Add existing colors to mapping
        if not overwrite_existing:
            for species in unique_species:
                for scheme in schemes.values():
                    if species in scheme["species"] and "color" in scheme["species"][species]:
                        color_mapping[species] = scheme["species"][species]["color"]
                        break

        for i, species in enumerate(unique_species):
            if not overwrite_existing and species in color_mapping:
                continue

            if method == "GR":
                hue += golden_ratio_conjugate + offset
            else:
                hue = hues[i] + offset
            hue %= 1

            np.random.seed(seed) # seed=None will try to read data from /dev/urandom (or the Windows analogue) if available or seed from the clock otherwise
            lightness = np.random.uniform(*lightness_range)
            np.random.seed(seed)
            saturation = np.random.uniform(*saturation_range)

            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
            color_mapping[species] = hex_color

        for scheme in schemes.values():
            for species in scheme["species"].keys():
                if overwrite_existing or "color" not in scheme["species"][species]:
                    scheme["species"][species]["color"] = color_mapping.get(species, scheme["species"][species].get("color"))

        if single_scheme:
            schemes = schemes['single_scheme']
        return schemes

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


