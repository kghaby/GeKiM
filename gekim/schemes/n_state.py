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
            Initialize a species object.

            Parameters
            ----------
            name : str
                Name of the species.
            y0 : Union[np.ndarray, float]
                Initial concentration of the species.
                Array Example: `{"Ligand":np.linspace(1,1500,100)}` for a Michaelis-Menten ligand concentration scan.
            label : str, optional
                Useful for plotting. Will default to NAME.
            color : str, optional
                Useful for plotting. Best added by ..utils.Plotting.assign_colors_to_species().

            """
            self.name = name
            self.y0 = np.array([y0]) if np.isscalar(y0) else np.array(y0)
            self.label = label or name
            self.index = None  # added by NState
            self.color = color
            self.sym = symbols(name)
            self.simin = None  # added by simulator
            self.simout = None  # added by simulator

    def __repr__(self):
        return f"{self.name} (Initial Concentration: {self.y0}, Label: {self.label})"

class Transition:
    def __init__(self, name: str, k, source: list, target: list, label=None, index=None):
        """
        Parameters
        ----------
        name : str
            Name of the rate constant.
        k : float
            Value of the rate constant.
        source : list
            List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings.
        target : list
            List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings.
        label : str, optional
            Could be useful for plotting. Will default to NAME.
        """
        self.name = name # should be the name of the rate constant for all intents and purposes, eg "kon"
        self.k = k
        self.source = Transition._format_state(source,"source")  # List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        self.target = Transition._format_state(target,"target")  # List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        self.label = label or name
        self.index = index
        self.sym = symbols(name)
        self.simin = None # added by simulator
        self.simout = None # added by simulator
        self.linear = self.is_linear()
        

    def __repr__(self):
        source_str = ' + '.join([f"{coeff}*{sp}" for sp, coeff in self.source])
        target_str = ' + '.join([f"{coeff}*{sp}" for sp, coeff in self.target])
        return f"{self.name} ({self.k}): {source_str} -> {target_str}"

    def is_linear(self):
        """
        Check if a transition is linear.

        Returns
        -------
        bool
            True if the transition is linear, False otherwise.
        """
        # A first-order reaction must have exactly one reactant with a stoichiometric coefficient of 1
        return len(self.source) == 1 and self.source[0][1] == 1

    @staticmethod
    def _parse_species_string(species_str):
        """
        Extract coefficient and species name from species string.

        Parameters
        ----------
        species_str : str
            A species string, e.g., '2A'.

        Returns
        -------
        tuple
            A tuple of species name (str) and stoichiometric coefficient (int).
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
    def _format_state(state,direction=None):
        """
        Format a transition by extracting and combining coefficients and species names.
        Is idempotent.

        Parameters
        ----------
        state : list
            State descriptor. List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings.
        direction : str, optional
            Direction of the transition. Default is None. Can be "source" or "target".

        Returns
        -------
        list
            List of (SPECIES, COEFF) tuples.

        Raises
        ------
        ValueError
            If the transition or species tuples are invalid.
        """
        parsed_species = {}
        for sp in state:
            if isinstance(sp, str):
                name, coeff = Transition._parse_species_string(sp)
            elif isinstance(sp, tuple):
                if len(sp) == 2:
                    if isinstance(sp[0], str) and isinstance(sp[1], (int, float)):
                        name, coeff = sp
                    elif isinstance(sp[1], str) and isinstance(sp[2], (int, float)):
                        coeff, name = sp
                    else:
                        raise ValueError(f"Invalid species tuple '{sp}' in transition '{state}'.")
                else:
                    raise ValueError(f"Invalid species tuple '{sp}' in transition '{state}'.")
            else:
                raise ValueError(f"Invalid species '{sp}' in transition '{state}'.")
            if direction == "source" and coeff < 0:
                raise ValueError(f"Negative coefficient '{coeff}' in source of transition '{state}'.")
            if name in parsed_species:
                parsed_species[name] += coeff # combine coeffs
            else:
                parsed_species[name] = coeff
        state = [(name, coeff) for name, coeff in parsed_species.items()]
        return state
    
class Path:
    """
    Represents a path in a network of species transitions. 
    Is created by `NState.find_paths()`

    Attributes
    ----------
    species_path : list
        List of species objects representing the path.
    transitions_path : list
        List of transition objects representing the transitions along the path.
    probability : float
        Probability of the path relative to other paths from `species[0]` to `species[-1]`

    Methods
    -------
    __repr__()
        Returns a string representation of the species path.

    """

    def __init__(self, species_path, transitions_path, probability):
        self.species_path = species_path
        self.transitions_path = transitions_path
        self.probability = probability

    def __repr__(self):
        """
        Returns a string representation of the Path object.

        Returns
        -------
        str
            String representation of the Path object.

        """
        path_str = ' -> '.join(['+'.join([sp.name for sp in group]) if isinstance(group, list) else group.name for group in self.species_path])
        return f"Path({path_str}, Probability: {self.probability})"

class NState:
    #TODO: Add stochastic method
    #TODO: add_species and add_transition methods
    #TODO: markovian, nonmarkovian, etc
    #TODO: is_linear() or get degree of linearity. will loops show up as a degree?
    
    def __init__(self, config: dict, logfilename=None, quiet=False):
            """
            Initialize the NState class with configuration data. Can be any degree of nonlinearity.

            Parameters
            ----------
            config : dict
                Configuration containing species and transitions.
                Species should contain name, initial concentration, and label.
                Transitions should contain name, source-species, target-species, and k value.
            logfilename : str, optional
                Name of the log file (default is None).
            quiet : bool, optional
                Flag indicating whether to suppress log output (default is False).

            Raises
            ------
            ValueError
                If config is invalid.
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
        Reinitialize the system after adding transitions or species.
        This method should be called if you modify the scheme after initialization.
        WARNING: This will reinitialize everything except the logger and config.
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

        Parameters
        ----------
        simulator : class
            The simulator class to use for the system. Unless using a custom simulator, 
            use the provided simulators in gekim.simulators.
        *args : tuple, optional
            Additional arguments to pass to the simulator.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the simulator.

        Notes
        -----
        This method is not as good as just doing:
        ```python
        system.simulator = simulator(system)
        system.simulator.simulate(...)
        ```
        because IDE syntax and doc helpers may not pick up the new simulator attribute and simulate method.
        """
        self.simulator = simulator(self, *args, **kwargs)
        self.simulate = self.simulator.simulate
        self.log.info(f"Simulator set to {simulator.__name__}.\n")
        self.log.info(f"Use system.simulator.simulate() or system.simulate() to run the simulation.\n")

    def sum_species_simout(self,whitelist:list=None,blacklist:list=None):
        """
        Sum the simout y-values of specified species.

        Parameters
        ----------
        whitelist : list, optional
            Names of species to include in the sum.
        blacklist : list, optional
            Names of species to exclude from the sum.

        Returns
        -------
        numpy.ndarray or None
            The sum of the simulated values. Returns None if the 
            simulated data is not found for any species.

        Raises
        ------
        ValueError
            If both whitelist and blacklist are provided.

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
        Save species vectors from a concentration matrix to the respective 
        `species[NAME].simout[key_name]` dict based on `species[NAME].index`.
        
        Parameters
        ----------
        matrix : numpy.ndarray
            The concentration matrix containing the species vectors.
        key_name : str, optional
            The key name to use for saving the species vectors in the species dictionary (default is "y").

        Notes
        -----
        Useful for saving the output of a continuous solution to the species dictionary.
        Don't forget to save time, too, eg `system.simout["t_cont"] = t`
        """
        for _, sp_data in self.species.items():
            sp_data.simout[key_name] = matrix[sp_data.index]
        return

    def find_paths(self, start_species: Union[str, Species], end_species: Union[str, Species], only_linear_paths=True, prob_cutoff=1e-3, max_depth=10):
        """
        Find paths from start_species to end_species.

        Parameters
        ----------
        start_species : str or Species
            Name or object of the starting species.
        end_species : str or Species
            Name or object of the ending species.
        only_linear_paths : bool, optional
            Whether to only find linear paths (no backtracking or loops) (default is True).
        prob_cutoff : float, optional
            Cutoff probability to stop searching current path (default is 1e-3).
        max_depth : int, optional
            Maximum depth to limit the search (default is 10).

        Returns
        -------
        list
            List of Path objects representing the found paths.
        """
        def get_transition_probability(transition, current_sp_name):
            # Get total rate for outgoing transitions from current_species
            total_rate = sum(tr.k for tr in self.transitions.values() if current_sp_name in [sp[0] for sp in tr.source])
            return transition.k / total_rate if total_rate > 0 else 0

        def dfs(current_sp_name, target_sp_name, visited_names, current_path, current_transitions, current_prob, depth):
            if current_sp_name == target_sp_name:
                self.paths.append(Path(current_path[:], current_transitions[:], current_prob))
                return

            if current_prob < prob_cutoff or depth > max_depth:
                return

            for transition in self.transitions.values():
                if current_sp_name in [sp[0] for sp in transition.source]:
                    next_species_list = [sp[0] for sp in transition.target]
                    if only_linear_paths and any(sp in visited_names for sp in next_species_list):
                        continue

                    for next_sp_name in next_species_list:
                        next_prob = current_prob * get_transition_probability(transition, current_sp_name)
                        visited_names.add(next_sp_name)
                        current_path.append(self.species[next_sp_name])
                        current_transitions.append(transition)
                        dfs(next_sp_name, target_sp_name, visited_names, current_path, current_transitions, next_prob, depth + 1)
                        #print(visited_names)
                        #visited_names.remove(next_sp_name)
                        current_path.pop()
                        current_transitions.pop()

        # Input validation
        all_linear_tr = True
        for transition in self.transitions.values():
            if not transition.linear:
                all_linear_tr = False
                self.log.warning(f"Transition '{transition.name}' is not linear!")
        if not all_linear_tr:
            self.log.warning("This method only uses TRANSITION.k to calculate probabilities, so they will likely be inaccurate.\n" +
                             "If possible, make all transitions linear (e.g., with a pseudo-first-order approximation).\n")

        if isinstance(start_species, str):
            start_species = self.species[start_species]
        elif isinstance(start_species, Species):
            pass
        else:
            raise ValueError("start_species must be a string or Species object.")
    
        if isinstance(end_species, str):
            end_species = self.species[end_species]
        elif isinstance(end_species, Species):
            pass
        else:
            raise ValueError("end_species must be a string or Species object.")

        # Search
        self.paths = []
        dfs(start_species.name, end_species.name, {start_species.name}, [start_species], [], 1.0, 0)

        self.log.info(f"Paths found from '{start_species.name}' to '{end_species.name}':")
        for path in self.paths:
            self.log.info(str(path))
        
        return self.paths