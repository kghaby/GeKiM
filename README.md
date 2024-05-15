# GeKiM (Generalized Kinetic Modeler)

## Description
GeKiM (Generalized Kinetic Modeler) is a Python package designed for creating, interpreting, and modeling arbitrary kinetic schemes with a focus on covalent inhibition. Schemes are defined by the user in a dictionary of species and transitions. These are then used to create instances of the NState class, which include methods of simulating and analyzing itself. 

The package also contains classes for common schemes, which come with scheme-specific analyses and metrics (e.g., ThreeState.KI, AXD.jacobian).

## Installation
With pip:
```bash
pip install gekim
```

Or directly from the source code:
```bash
git clone https://github.com/kghaby/GeKiM.git
cd GeKiM
pip install .
```

## Usage
Here is a basic example of how to use GeKiM to create and simulate a kinetic model:
```python
import gekim as gk
from gekim.analysis import covalent_inhibition as ci

# Define your kinetic scheme in a configuration dictionary
concI0,concE0 = 100,1
scheme = {
    'species': {
        "I": {"y0": concI0, "label": "I"},
        "E": {"y0": concE0, "label": "E"},
        "EI": {"y0": 0, "label": "EI"},
    },    
    'transitions': {
        "kon": {"k": 0.01, "source": ["2E","I"], "target": ["EI"]},
        "koff": {"k": 0.1, "source": ["EI"], "target": ["2E","I"]},
    }
}

# Create a model
system = gk.schemes.NState(scheme)

# Choose a simulator and go. In this example we're doing a deterministic simulation of the concentrations of each species. 
system.simulator = gk.simulators.ODESolver(system)
system.simulator.simulate()

# Fit the data to experimental models to extract mock-experimental measurements
final_state = system.species["EI"].simout["y"]
all_bound = system.sum_species_simout(blacklist=["E","I"])
fit_output = ci.kobs_uplim_fit_to_occ_final_wrt_t(
    t,final_state,nondefault_params={"Etot":{"fix":concE0}})
print(f"Fit: {fit_output.fitted_params}\n")
```
For more detailed examples, please refer to the examples directory.

## Documentation
API Documentation with examples will be done eventually.

## Contributing
If you have suggestions or want to contribute code, please feel free to open an issue or a pull request.

## License
GeKiM is licensed under the GPL-3.0 license.

## Contact
kyleghaby@gmail.com

## TODO
so much