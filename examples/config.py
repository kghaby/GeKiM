# Example config
import gekim as gk
Ki = 10 #nM, koff/kon
koff = 1/(15*60) #1/15 min converted to s #0.00111111111
kon = koff/Ki
concI0,concE0=100,1
kinactf = 0.01
kinactb = 0.0001

schemes = {}
schemes["3S"] = {
    "transitions": {
        "kon": {"k": kon, "source": ["E", "I"], "target": ["E_I"], "label": r"$k_{on}$"},
        "koff": {"k": koff, "source": ["E_I"], "target": ["E", "I"], "label": r"$k_{off}$"},
        "kinactf": {"k": kinactf, "source": ["E_I"], "target": ["EI"]}, #irrev step
        "kinactb": {"k": kinactb, "source": ["EI"], "target": ["E_I"]},
    },
    "species": {
        "I": {"y0": concI0, "label": r"I"},
        "E": {"y0": concE0, "label": r"E"},
        "E_I": {"y0": 0, "label": r"E${\cdot}$I"},
        "EI": {"y0": 0, "label": r"E${\mydash}$I"},
    },
}

schemes = gk.utils.plotting.assign_colors_to_species(schemes,saturation_range=(0.5,0.8),lightness_range=(0.4,0.4),offset=0.0,method="GR",overwrite_existing=False)