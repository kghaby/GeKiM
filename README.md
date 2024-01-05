# GeKiM (Generalized Kinetic Modeler)

## Overview
GeKiM (Generalized Kinetic Modeler) is a Python package designed for modeling arbitrary kinetic schemes with a focus on biochemical systems. It provides a flexible framework for simulating and analyzing the kinetics of complex reactions, enabling researchers and scientists to gain insights into the dynamics of biochemical processes.

## Features
- **Versatile Kinetic Modeling:** Allows the definition of a wide range of kinetic models, supporting various types of biochemical reactions.
- **Robust ODE Solving:** Utilizes efficient numerical methods to solve ordinary differential equations (ODEs) arising from kinetic models.
- **Intuitive Configuration:** Kinetic schemes and species can be easily configured using a simple, user-friendly format.
- **Detailed Logging and Analysis:** Provides capabilities for detailed logging of simulation processes and comprehensive analysis of results.

## Installation
For now, you can only install GeKiM directly from the source code:
```bash
git clone https://github.com/kghaby/GeKiM.git
cd GeKiM
pip install .
```

## Usage
Here is a basic example of how to use GeKiM to create and simulate a kinetic model:
```python
from GeKiM import NState

# Define your kinetic scheme in a configuration dictionary
config = {
    'species': {
        "I": {"conc": 100, "label": "$I$"},
        "E": {"conc": 1, "label": "$E$"},
        "EI": {"conc": 0, "label": "$EI$"},
    },    
    'transitions': {
        "kon": {"value": 0.0001, "from": ["E","I"], "to": ["EI"]},
        "koff": {"value": 0.01, "from": ["EI"], "to": ["E","I"]},
    }
}

# Create a model
model = NState(config)

# Define time points and simulate. In this example we're doing a deterministic simulation of the concentrations of each species. 
time_points = [0, 10, 20, 30] 
model.solve_ode(time_points)

# Solution will be columned data of concentrations
print(model.sol)
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