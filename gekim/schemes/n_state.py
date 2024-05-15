import numpy as np
import re
import copy
from sympy import symbols
from typing import Union
from ..utils.helpers import integerable_float
from ..utils.logging import Logger



#TODO: find_linear_paths in kinetics2 and more general pathfinder in utils?
    
class Species:
    def __init__(self, name: str, y0: Union[np.ndarray,float], label=None, color=None):
        """
        Recommended to initialize with the following arguments:
        name: Name of the species.
        y0: Initial concentration of the species.
            Array Example: {"Ligand":np.linspace(1,1500,100)} for a Michaelis-Menten ligand concentration scan.
        label: Useful for plotting. Will default to NAME.
        color: useful for plotting. Best added by ..utils.Plotting.assign_colors_to_species().

        """
        self.name = name
        self.y0 = np.array([y0]) if np.isscalar(y0) else np.array(y0)
        self.label = label or name
        self.index = None # added by NState
        self.color = color
        self.sym = symbols(name)
        self.simin = None # added by simulator
        self.simout = None # added by simulator

    def __repr__(self):
        return f"{self.name} (Initial Concentration: {self.y0}, Label: {self.label})"

class Transition:
    def __init__(self, name: str, k, source: list, target: list, label=None, index=None):
        """
        Recommended to initialize with the following arguments:
        name: Name of the rate constant.
        k: Value of the rate constant.
        source: List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        target: List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        label: Could be useful for plotting. Will default to NAME.

        """
        self.name = name # should be the name of the rate constant for all intents and purposes, eg "kon"
        self.k = k
        self.source = Transition._format_transition(source,"source")  # List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        self.target = Transition._format_transition(target,"target")  # List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        self.label = label or name
        self.index = index
        self.sym = symbols(name)
        self.simin = None # added by simulator
        self.simout = None # added by simulator
        

    def __repr__(self):
        source_str = ' + '.join([f"{coeff}*{sp}" for sp, coeff in self.source])
        target_str = ' + '.join([f"{coeff}*{sp}" for sp, coeff in self.target])
        return f"{self.name} ({self.k}): {source_str} -> {target_str}"

    @staticmethod
    def _parse_species_string(species_str):
        """
        Extract coefficient and species name from species string.

        Args:
        species_str (str): A species string, e.g., '2A'.

        Returns:
        tuple: A tuple of species name (str) and stoichiometric coefficient (int).
        """
        match = re.match(r"(-?\d*\.?\d*)(\D.*)", species_str)
        if match and match.groups()[0]:
            coeff = match.groups()[0]
            if coeff == '-':
                coeff = -1
            coeff = integerable_float(float(coeff))
        else:
            coeff = 1
        name = match.groups()[1] if match else species_str
        return name,coeff
    
    @staticmethod            
    def _format_transition(tr,direction=None):
        """
        Format a transition by extracting and combining coefficients and species names.
        Is idempotent.
        """
        parsed_species = {}
        for sp in tr:
            if isinstance(sp, str):
                name, coeff = Transition._parse_species_string(sp)
            elif isinstance(sp, tuple):
                if len(sp) == 2:
                    if isinstance(sp[0], str) and isinstance(sp[1], (int, float)):
                        name, coeff = sp
                    elif isinstance(sp[1], str) and isinstance(sp[2], (int, float)):
                        coeff, name = sp
                    else:
                        raise ValueError(f"Invalid species tuple '{sp}' in transition '{tr}'.")
                else:
                    raise ValueError(f"Invalid species tuple '{sp}' in transition '{tr}'.")
            else:
                raise ValueError(f"Invalid species '{sp}' in transition '{tr}'.")
            if direction == "source" and coeff < 0:
                raise ValueError(f"Negative coefficient '{coeff}' in source of transition '{tr}'.")
            if name in parsed_species:
                parsed_species[name] += coeff # combine coeffs
            else:
                parsed_species[name] = coeff
        tr = [(name, coeff) for name, coeff in parsed_species.items()]
        return tr

class NState:
    #TODO: Add stochastic method
    #TODO: add_species and add_transition methods
    #TODO: markovian, nonmarkovian, etc
    #TODO: is_linear() or get degree of linearity. will loops show up as a degree?
    
    def __init__(self, config: dict, logfilename=None, quiet=False):
        """
        Initialize the NState class with configuration data. Can be any degree of nonlinearity.

        Args:
        config (dict): Configuration containing species and transitions.
                       Species should contain name, initial concentration, and label.
                       Transitions should contain name, source-species, target-species, value, and label.

        Raises:
        ValueError: If config is invalid.
        """
        self.log = Logger(quiet=quiet, logfilename=logfilename)

        self._validate_config(config)
        self.config = copy.deepcopy(config)
    
        self.species = {
            name: Species(
                name=name,
                y0=np.array([data["y0"]]) if np.isscalar(data["y0"]) else np.array(data["y0"]),
                label=data.get('label', name),
                color=data.get('color')
            ) for name, data in config['species'].items()
        }

        self.transitions = {
            name: Transition(
                name=name,
                k=data['k'],
                source=data['source'],
                target=data['target'],
                label=data.get('label', name)
            ) for name, data in config['transitions'].items()
        }

        self.setup()
        self.simulator = None
        self.log.info(f"NState system initialized successfully.\n")

    def setup(self):
        """
        Use this if you added transitions or species after initialization.
        This is called in __init__ so you don't need to call it again unless you change the scheme.
        WARNING: This will basically reinitialize everything besides the logger and config.
        """
        #TODO: this needs testing. Make sure that its fine to not reinit the concentrations
        # Document the order of the species
        for idx, name in enumerate(self.species):
            self.species[name].index = idx
        self._validate_species()
            
        # Document the order of the transitions
        for idx, name in enumerate(self.transitions):
            self.transitions[name].index = idx

        return

    def _validate_config(self,config):
        if not 'species' in config or not 'transitions' in config:
            raise ValueError("Config must contain 'species' and 'transitions' keys.")
        return True

    def _validate_species(self):
        """
        Validate the species data in the configuration.

        Returns:
        bool: True if valid, False otherwise.
        """
        #TODO: use assign color 
        labels = set()
        for name, data in self.species.items():
            # Validate labels
            label = data.label
            if label in labels:
                self.log.error(f"Duplicate label '{label}' found for species '{name}'.")
                return False
            labels.add(label)
        return True

    def set_simulator(self, simulator, *args, **kwargs) -> None:
        """
        Sets and initializes the simulator for the system. 
        Sets the document string of the simulate method to that of the simulator.
        
        This isn't as good as just doing 
            ```python
            system.simulator = simulator(system)
            system.simulator.simulate(...)
            ```
        because IDE syntax and doc helpers can't seem to pick up the new simulator attribute and simulate method.

        Args:
        - simulator: Unless using a custom simulator, use the provided simulators in gekim.simulators.
        - *args: Additional arguments to pass to the simulator.
        - **kwargs: Additional keyword arguments to pass to the simulator.
        """
        self.simulator = simulator(self, *args, **kwargs)
        self.simulate = self.simulator.simulate
        self.log.info(f"Simulator set to {simulator.__name__}.\n")
        self.log.info(f"Use system.simulator.simulate() or system.simulate() to run the simulation.\n")

    def sum_species_simout(self,whitelist:list=None,blacklist:list=None):
        """
        Sum the simout y-values of specified species.
        Whitelist and blacklist cannot be provided simultaneously.

        Args:
        - whitelist: Names of species to include in the sum.
        - blacklist: Names of species to exclude from the sum.

        Returns:
        - The sum of the simulated values.
        """
        if whitelist and blacklist:
            raise ValueError("Provide either a whitelist or a blacklist, not both.")

        species_names = self.species.keys()
        
        if whitelist:
            species_names = [name for name in whitelist if name in species_names]
        elif blacklist:
            species_names = [name for name in species_names if name not in blacklist]

        try:
            total_y = np.sum([self.species[name].simout["y"] for name in species_names], axis=0)
        except KeyError as e:
            self.log.error(f"Simulated data not found for species '{e}'.")
            return None
        return total_y

    def mat2sp_simout(self,matrix,key_name="y"):
        """
        Save species vectors from a concentration matrix to the respective species[NAME].simout[key_name] dict based on species[NAME].index.
        Useful for saving the output of a continuous solution to the species dictionary.
            Don't forget `system.simout["t_cont"] = t`
        """
        for _, sp_data in self.species.items():
            sp_data.simout[key_name] = matrix[sp_data.index]
        return

                


