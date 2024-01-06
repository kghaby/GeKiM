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

t = np.linspace(0.0001, 10000, 10000)

model = gekim.NState(schemes["3S"])
model,kobs=gekim.utils.solveModel(t,model,"CO")

fig = plt.figure(figsize=(5, 3))
plt.title("3S")
plt.plot(t, np.sum(model.sol[:, 2:], axis=1),label='All Bound States',color="grey")
for species, props in model.species.items():
    if species == "I":
        continue
    plt.plot(t, model.sol[:, model.species_order[species]], label=props['label'],color=schemes["3S"]["species"][species]["color"])

plt.plot(t, gekim.utils.occFromKobs(t,kobs,schemes["3S"]["species"]["E"]["conc"]),label=r"$k_{\text{obs}}$ = "+str(gekim.utils.round_sig(kobs,3))+r" $\text{s}^{-1}$",ls='--', color="black")

plt.xlabel('Time (s)')
plt.ylabel('Concentration (nM)')
plt.legend(frameon=False)

plt.show()