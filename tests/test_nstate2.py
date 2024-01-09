import gekim
import numpy as np
import matplotlib.pyplot as plt

schemes={}
schemes["3S"] = {
    "species": {
        "I": {"conc": 100, "label": r"$I$"},
        "E": {"conc": 1, "label": r"$E$"},
        "E_I": {"conc": 0, "label": r"$E{\cdot}I$"},
        "EI": {"conc": 0, "label": r"$E\text{-}I$"},
    },
    "transitions": {
        "k1": {"value": 0.0001, "from": ["E", "I"], "to": ["E_I"], "label": r"$k_{on}$"},
        "k2": {"value": 0.01, "from": ["E_I"], "to": ["E", "I"], "label": r"$k_{off}$"},
        "k3": {"value": 0.001, "from": ["E_I"], "to": ["EI"]}, #irrev step
        "k4": {"value": 0, "from": ["EI"], "to": ["E_I"]},
    },
}
schemes["3Scoeff1"] = {
    "species": {
        "I": {"conc": 100, "label": r"$I$"},
        "E": {"conc": 1, "label": r"$E$"},
        "E_I": {"conc": 0, "label": r"$E{\cdot}I$"},
        "EI": {"conc": 0, "label": r"$E\text{-}I$"},
    },
    "transitions": {
        "k1": {"value": 0.0001, "from": ["E", "2I"], "to": ["E_I"], "label": r"$k_{on}$"},
        "k2": {"value": 0.01, "from": ["E_I"], "to": ["E", "2I"], "label": r"$k_{off}$"},
        "k3": {"value": 0.001, "from": ["3E_I"], "to": ["7EI"]}, #irrev step
        "k4": {"value": 0, "from": ["7EI"], "to": ["3E_I"]},
    },
}
schemes["3Scoeff2"] = {
    "species": {
        "I": {"conc": 100, "label": r"$I$"},
        "E": {"conc": 1, "label": r"$E$"},
        "E_I": {"conc": 0, "label": r"$E{\cdot}I$"},
        "EI": {"conc": 0, "label": r"$E\text{-}I$"},
    },
    "transitions": {
        "k1": {"value": 0.0001, "from": ["E", "I", "I"], "to": ["E_I"], "label": r"$k_{on}$"},
        "k2": {"value": 0.01, "from": ["E_I"], "to": ["E", "I", "I"], "label": r"$k_{off}$"},
        "k3": {"value": 0.001, "from": ["E_I","E_I","E_I"], "to": ["7EI"]}, #irrev step
        "k4": {"value": 0, "from": ["7EI"], "to": ["E_I","E_I","E_I"]},
    },
}
schemes=gekim.utils.assign_colors_to_species(schemes,saturation_range=(0.5,0.8),lightness_range=(0.4,0.5),overwrite_existing=False)

t = np.linspace(0.0001, 10000, 3)

model = gekim.NState(schemes["3S"],quiet=True)
model.simulate_deterministic(t)
print(model.traj_deterministic)
#model,kobs=gekim.utils.solve_model(t,model,"CO")
