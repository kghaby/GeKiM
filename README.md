# GeKiM (Generalized Kinetic Modeler)

## Description
GeKiM (Generalized Kinetic Modeler) is a Python package designed for creating, interpreting, and modeling arbitrary kinetic schemes. Schemes are defined by the user in a dictionary of species and transitions, which is used to initialize an instance of the NState class. Choose (or make) and initialize a simulator for the instance and run it. Field-specific practices are found in gekim/fields.

## Installation
With pip:
```bash
pip install gekim
```

Or directly from the source code (recommended):
```bash
git clone https://github.com/kghaby/GeKiM.git
cd GeKiM
pip install .
```

## Usage
Here is a basic example of how to use GeKiM to create and simulate a kinetic system:
```python
import gekim as gk
from gekim.fields.bio.enzyme.inhib import irrev as ii 

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

# Initialize a system with your schematic dictionary
system = gk.schemes.NState(scheme,quiet=False)

# Choose a simulator and go. In this example we're doing a deterministic 
# simulation of the concentrations of each species over time.
# Note that `system.simulator() = gk.simulators.ODESolver(system)` may be more doc-hint friendly
system.set_simulator(gk.simulators.ODESolver)
system.simulator.simulate() 

# Fit the data to experimental models to extract mock-experimental measurements
final_state = system.species["EI"].simout["y"]
all_bound = system.sum_species_simout(blacklist=["E","I"])

fit_output = ii.kobs_uplim_fit_to_occ_final_wrt_t(
    t,final_state,nondefault_params={"Etot":{"value":concE0,"vary":False}})

```
For more detailed examples, please refer to the examples directory.

## Documentation
Documentation and example notebook(s) are pending.

## Contributing
If you have suggestions or want to contribute code, please feel free to open an issue or a pull request.

## License
GeKiM is licensed under the GPL-3.0.

## Contact
kyleghaby@gmail.com
