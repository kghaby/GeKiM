# Example config
import gekim
Ki = 10 #nM, koff/kon
koff = 1/(15*60) #1/15 min converted to s #0.00111111111
kon = koff/Ki
concI0,concE0=100,1
kinactf = 0.01
kinactb = 0.00#00001

schemes = {}
schemes["Three State"] = {
    "transitions": {
        "kon": {"value": kon, "from": ["E","E", "I"], "to": ["E_I"], "label": r"$k_{on}$"},
        "koff": {"value": koff, "from": ["E_I"], "to": ["3E", "I"], "label": r"$k_{off}$"},
        "kinactf": {"value": kinactf, "from": ["E_I"], "to": ["EI"]}, #irrev step
        "kinactb": {"value": kinactb, "from": ["EI"], "to": ["E_I"]},
    },
    "species": {
        "I": {"conc": concI0, "label": r"I"},
        "E": {"conc": concE0, "label": r"E"},
        "E_I": {"conc": 0, "label": r"E${\cdot}$I"},
        "EI": {"conc": 0, "label": r"E${\mydash}$I"},
    },
}
schemes = gekim.utils.Plotting.assign_colors_to_species(schemes,saturation_range=(0.5,0.8),lightness_range=(0.4,0.4),offset=0.0,method="GR",overwrite_existing=False)