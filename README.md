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
import gekim

# Define your kinetic scheme in a configuration dictionary
config = {
    'species': {
        "I": {"conc": 100, "label": "$I$"},
        "E": {"conc": 1, "label": "$E$"},
        "EI": {"conc": 0, "label": "$EI$"},
    },    
    'transitions': {
        "kon": {"value": 0.01, "from": ["E","I"], "to": ["EI"]},
        "koff": {"value": 0.1, "from": ["EI"], "to": ["E","I"]},
    }
}

# Create a model
model = gekim.scheme.NState(config)

# Define time points and simulate. In this example we're doing a deterministic simulation of the concentrations of each species. 
t = np.linspace(0.0001, 1000, 10)
model.simulate_deterministic(t)

# Solution will be columned data of concentrations
print(model.traj_deterministic)

# Find the observed rate of occupancy 
occupancy = model.traj_deterministic[:,model.species_order["EI"]]
kobs = gekim.analysis.CovalentInhibition.kobs_fit_to_occ_final_wrt_t(t,occupancy)
print(kobs)
```
For more detailed examples, please refer to the examples directory.

## Documentation
API Documentation with examples can be found at TODO.

## Contributing
If you have suggestions or want to contribute code, please feel free to open an issue or a pull request.

## License
GeKiM is licensed under the GPL-3.0 license.

## Contact
kyleghaby@gmail.com

## TODO
