import colorsys
import numpy as np 
from matplotlib.colors import LinearSegmentedColormap
from colorsys import rgb_to_hsv, hsv_to_rgb

def assign_colors_to_species(schemes: dict, saturation_range: tuple = (0.5, 0.7), lightness_range: tuple = (0.3, 0.4), 
                            method: str = None, offset: float = 0, overwrite_existing=False, seed: int = None):
    """
    Assigns a distinct and aesthetically pleasing color to each species in a dictionary or a single kinetic scheme.
    Uses either a fixed or golden ratio based distribution for hues. Optionally seeds the randomness for consistent results.

    Parameters
    ----------
    schemes : dict
        Dictionary of kinetic scheme dictionaries or a single kinetic scheme dictionary.
    saturation_range : tuple
        Min and max saturation values.
    lightness_range : tuple
        Min and max lightness values.
    method : str
        "GR" for golden ratio hue distribution; None for linear distribution.
    offset : float
        Offset value for the hues.
    overwrite_existing : bool
        If True, overwrite existing colors; if False, assign colors only to species without colors.
    seed : int 
        Seed for random number generator for reproducible color variations.

    Returns
    -------
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

def scale_cmap_saturation(cmap: LinearSegmentedColormap, scalar:float = 1.5) -> LinearSegmentedColormap:
    """
    Scale the saturation of a matplotlib colormap. 
    
    Returns
    -------
    LinearSegmentedColormap: Scaled colormap named `cmap.name`+'scaledsat'
    """
    colors = cmap(np.linspace(0, 1, cmap.N)) 
    hsv = np.array([rgb_to_hsv(*color[:3]) for color in colors]) # convert to hsv
    hsv[:, 1] = np.clip(hsv[:, 1] * scalar, 0, 1)
    new_colors = np.array([hsv_to_rgb(*hsv_color) for hsv_color in hsv]) # convert back to rgb
    return LinearSegmentedColormap.from_list(f"{cmap.name}_scaledsat", new_colors)

    
