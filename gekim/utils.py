from scipy.optimize import curve_fit
import numpy as np
import math
import colorsys

def rate_pair_from_P(intOOM,Pf):
    """
    Provides rates between two states that are at rapid equilibrium and therefore approximated by the population distribution. 
    intOOM is the order of magnitude of the rates.
    Pf is the probability of the forward state, ie kf/(kf+kb) or [B]/([A]+[B]) for A<-->B.
    """
    kf=Pf*10**(float(intOOM)+1)
    kb=10**(float(intOOM)+1)-kf
    return kf,kb

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
    def assign_colors_to_species(schemes, saturation_range=(0.5, 0.7), lightness_range=(0.3, 0.4), method=None, offset=0, overwrite_existing=False):
        """
        Assigns a distinct and aesthetically distributed color to each species in the provided dictionary of kinetic schemes. 
        Uses the golden ratio for even hue distribution.
        Method can be "GR" to use the golden ratio to generate the hues.
        Offset can be used to offset the hues.
        overwrite_existing: If True, overwrite existing colors; if False, assign colors only to species without colors.
        """
        #TODO: This only works with a dictionary of schemes. It should work on single schemes too
        unique_species = set()
        for scheme in schemes.values():
            unique_species.update(scheme["species"].keys())

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
                        break # Keep first match
                
        for i, species in enumerate(unique_species):
            # Skip species with existing color unless overwriting
            if not overwrite_existing and species in color_mapping:
                continue

            if method == "GR":
                hue += golden_ratio_conjugate + offset
            else:
                hue = hues[i] + offset
            hue %= 1

            lightness = np.random.uniform(*lightness_range)
            saturation = np.random.uniform(*saturation_range)
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            hex_color = f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'
            color_mapping[species] = hex_color

        for scheme in schemes.values():
            for species in scheme["species"].keys():
                if overwrite_existing or "color" not in scheme["species"][species]:
                    scheme["species"][species]["color"] = color_mapping.get(species, scheme["species"][species].get("color"))

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


