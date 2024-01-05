## Example config
Ki = 1 #nM, koff/kon
koff = 1/(15*60) #1/15 min converted to s 
conc0I,conc0E=3430,1
schemes = {}
kf,kb=makeIntRates(6,0.1) #microsecond intOOM = 6 
schemes["S2"] = {
    "transitions": {
        "k1": {"value": koff/Ki, "from": ["E", "I"], "to": ["E___I"], "label": r"$k_{on}$"},
        "k2": {"value": koff, "from": ["E___I"], "to": ["E", "I"], "label": r"$k_{off}$"},
        "k_ncf": {"value": kf, "from": ["E___I"], "to": ["noncovalent_good"]},
        "k_ncb": {"value": kb, "from": ["noncovalent_good"], "to": ["E___I"]},
        "k_nc-t": {"value": 1.8457E+04, "from": ["noncovalent_good"], "to": ["thiolate"]},
        "k_t-nc": {"value": 2.25220018E+11, "from": ["thiolate"], "to": ["noncovalent_good"]},
        "k3": {"value": 4.4152E+08, "from": ["thiolate"], "to": ["E_I"]},
        "k4": {"value": 7.28238E+05, "from": ["E_I"], "to": ["thiolate"]},
        "k5": {"value": 2.2E+09, "from": ["E_I"], "to": ["E_I_"]}, #proton leaving
        "k6": {"value": 4.0e2, "from": ["E_I_"], "to": ["E_I"]}, #proton returning in pH 7.4
        "k7": {"value": 1.028E+10, "from": ["E_I"], "to": ["EI"]}, #irrev step
        "k8": {"value": 2.501E-06, "from": ["EI"], "to": ["E_I"]},
    },
    "species": {
        "I": {"conc": conc0I, "label": r"$I$"},
        "E": {"conc": conc0E, "label": r"$E$"},
        "E___I": {"conc": 0, "label": r"$E{\mydot\mydot\mydot}I$"},
        "noncovalent_good": {"conc": 0, "label": r"$E{\mydot\mydot}I$"},
        "thiolate": {"conc": 0, "label": r"$E_{(thiolate)}^{-}{\mydot\mydot}I+H_{(site)}^{+}$"},
        "E_I": {"conc": 0, "label": r"$E{\mydash}I_{(enolate)}^{-}+H_{(site)}^{+}$"},
        "E_I_": {"conc": 0, "label": r"$E{\mydash}I_{(enolate)}^{-}+H_{(bulk)}^{+}$"},
        "EI": {"conc": 0, "label": r"$E{\mydash}I$"},
    },
}
schemes=assign_colors_to_species(schemes,saturation_range=(0.5,0.8),lightness_range=(0.4,0.4),offset=0.0,method="GR",overwrite_existing=False)