import colorsys
import numpy as np 
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from colorsys import rgb_to_hsv, hsv_to_rgb
import gc
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..schemes.scheme import Scheme

def clear_fig(fig: plt.Figure):
    """Clear and garbage collect a matplotlib figure"""
    for ax in fig.get_axes():
        ax.clear()
    fig.clear()
    gc.collect()
    return

def assign_colors_to_species(schemes: Union[dict, "Scheme"], method: str = "preset1",
                             overwrite_existing=False,
                             saturation_range: tuple = (0.5, 0.7), 
                             lightness_range: tuple = (0.3, 0.4), 
                             offset: float = 0, seed: int = None):
    """
    Assign colors to species across one or more kinetic schemes.

    Parameters
    ----------
    schemes : dict or gekim.Scheme
        One of the following:
        - Single scheme config dictionary with keys ``{"species", "transitions"}``
        - Dictionary mapping names to scheme config dictionaries
        - Single :class:`gekim.Scheme` object
        - Dictionary mapping names to :class:`gekim.Scheme` objects

    method : {"preset", "GR", "lindist"}, optional
        Color assignment strategy.
        - ``"preset"`` : Cycle through a fixed, hard-coded hex color palette (default)
        - ``"GR"``      : Golden-ratio walk in hue space
        - ``"lindist"`` : Linear hue distribution in HLS space

    overwrite_existing : bool, optional
        If ``True``, overwrite existing species colors.
        If ``False``, assign colors only to species without a defined color.

    saturation_range : tuple of float, optional
        ``(min, max)`` saturation range for HLS-based methods (``"GR"``, ``"lindist"``).

    lightness_range : tuple of float, optional
        ``(min, max)`` lightness range for HLS-based methods (``"GR"``, ``"lindist"``).

    offset : float, optional
        Additive offset applied to hue values for HLS-based methods.

    seed : int or None, optional
        Seed for the random number generator controlling saturation/lightness
        sampling, enabling reproducible color assignments.

    Returns
    -------
    dict or gekim.Scheme
        Input schemes with species colors assigned in-place.
    """
    #TODO: support cmaps
        # if hasattr(cmap, "colors"):
        #     color = cmap.colors[i%len(cmap.colors)]
        # else:
        #     color = cmap(i / len(permutations))
    #TODO: xkcd method for cycling through xkcd colors
    
    preset1 = ['#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6', '#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6', '#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6','#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6', '#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6','#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6','#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9', '#e3b2ed', '#635c60', '#b2edb8', '#e8a0b6', '#3483eb', '#677d56', '#874ae8',  '#87c9ba', '#c95757', '#e69753', '#66bfd4', '#d466ad', '#879bc9', '#6672d4', '#d9c56f', '#6fd9a0', '#6fa0d9']


    def _is_scheme_obj(x) -> bool:
        return hasattr(x, "species") and hasattr(x, "transitions")

    def _is_scheme_cfg(x) -> bool:
        return isinstance(x, dict) and ("species" in x) and ("transitions" in x)

    def _iter_species_keys_from_obj(obj: "Scheme"):
        return obj.species.keys()

    def _iter_species_keys_from_cfg(cfg: dict):
        return cfg.get("species", {}).keys()

    def _get_existing_color_cfg(cfg: dict, sp: str):
        d = cfg["species"].get(sp, {})
        return d.get("color", None)

    def _set_color_cfg(cfg: dict, sp: str, color: str):
        cfg["species"].setdefault(sp, {})["color"] = color

    def _get_existing_color_obj(obj: "Scheme", sp: str):
        return getattr(obj.species[sp], "color", None)

    def _set_color_obj(obj: "Scheme", sp: str, color: str):
        obj.species[sp].color = color

    # normalize input -> list of (kind, ref) where kind in {"cfg","obj"}
    normalized = []

    if _is_scheme_cfg(schemes):
        normalized = [("cfg", schemes)]
    elif _is_scheme_obj(schemes):
        normalized = [("obj", schemes)]
    elif isinstance(schemes, dict):
        # dict of configs or dict of objects (may be mixed, but that's pathological; allow mixed anyway)
        for v in schemes.values():
            if _is_scheme_cfg(v):
                normalized.append(("cfg", v))
            elif _is_scheme_obj(v):
                normalized.append(("obj", v))
            else:
                raise ValueError("Dict values must be Scheme objects or scheme config dicts.")
        if not normalized:
            return schemes
    else:
        raise ValueError("Input must be a Scheme, a scheme config dict, or a dict of those.")

    # unique species in insertion order across all schemes
    unique_species = []
    seen = set()
    for kind, sch in normalized:
        keys = _iter_species_keys_from_cfg(sch) if kind == "cfg" else _iter_species_keys_from_obj(sch)
        for sp in keys:
            if sp not in seen:
                seen.add(sp)
                unique_species.append(sp)

    # initial mapping from pre-existing colors (first occurrence wins)
    color_mapping = {}
    if not overwrite_existing:
        for sp in unique_species:
            for kind, sch in normalized:
                if kind == "cfg":
                    if sp in sch["species"]:
                        c = _get_existing_color_cfg(sch, sp)
                    else:
                        c = None
                else:
                    if sp in sch.species:
                        c = _get_existing_color_obj(sch, sp)
                    else:
                        c = None
                if c:
                    color_mapping[sp] = c
                    break

    # assign new colors
    n = len(unique_species)
    if method is None:
        method = "preset1"
    if method not in {"preset1", "lindist", "GR"}:
        raise ValueError("method must be one of {'preset1','lindist','GR'}.")

    if method == "preset1":
        for i, sp in enumerate(unique_species):
            if not overwrite_existing and sp in color_mapping:
                continue
            color_mapping[sp] = preset1[i % len(preset1)]
    else:
        golden_ratio_conjugate = 0.618033988749895
        hues = np.linspace(0.0, 1.0, n, endpoint=False)
        hue = 0.0

        rng = np.random.default_rng(seed)
        for i, sp in enumerate(unique_species):
            if not overwrite_existing and sp in color_mapping:
                continue

            if method == "GR":
                hue = (hue + golden_ratio_conjugate + float(offset)) % 1.0
            else:  # "lindist"
                hue = (float(hues[i]) + float(offset)) % 1.0

            lightness = float(rng.uniform(*lightness_range))
            saturation = float(rng.uniform(*saturation_range))
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            color_mapping[sp] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    # write back into all schemes
    for kind, sch in normalized:
        if kind == "cfg":
            for sp in sch["species"].keys():
                if overwrite_existing or ("color" not in sch["species"][sp]) or (not sch["species"][sp].get("color")):
                    _set_color_cfg(sch, sp, color_mapping[sp])
        else:
            for sp in sch.species.keys():
                if overwrite_existing or (not _get_existing_color_obj(sch, sp)):
                    _set_color_obj(sch, sp, color_mapping[sp])

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

    
