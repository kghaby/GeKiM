import numpy as np
import re
from copy import deepcopy
from sympy import symbols
from collections import defaultdict
from typing import Union
from ..utils.helpers import integerable_float
from ..utils.logging import Logger
#todo: dataclasses for species transitions and paths 
class Species:
    def __init__(self, name: str, y0: Union[np.ndarray,float], label=None, color=None):
            """
            Initialize a species object.

            Parameters
            ----------
            name : str
                Name of the species.
            y0 : Union[np.ndarray,float]
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

        Notes
        -----
        A first-order reaction must have exactly one source species (with a stoichiometric coefficient of 1).
        """
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
    length : int
        Length of species_path.

    Methods
    -------
    __repr__()
        Returns a string representation of the species path.

    """

    def __init__(self, species_path, transitions_path, probability):
        self.species_path = species_path
        self.transitions_path = transitions_path
        self.probability = probability
        self.length = len(species_path)

    def __repr__(self):
        """
        Returns a string representation of the Path object.

        Returns
        -------
        str
            String representation of the Path object.

        """
        path_str = ' -> '.join(['+'.join([sp.name for sp in group]) if isinstance(group, list) else group.name for group in self.species_path])
        prob_fmt = "{:.2e}".format(self.probability)
        return f"Path(Length: {str(self.length).rjust(3)},\tProbability: {prob_fmt},\t{path_str})"

class NState:
    #TODO: add/remove_species and add/remove_transition methods
    #TODO: reinits if species or transitions are modified. let user able to define custom rate law funcs 
    #TODO: markovian, nonmarkovian, etc
    #TODO: add a function for fetching info. say get_simout("species_name")

    
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
            self.config = deepcopy(config)
        
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
            self.paths = None
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
    
    def simulate(self, simulator=None, *args, **kwargs):
        """
        Simulate the system using the provided simulator.

        Parameters
        ----------
        simulator : class, optional
            The simulator class to use for the system. Unless using a custom simulator, 
            use the provided simulators in gekim.simulators.
        *args : tuple, optional
            Additional arguments to pass to the simulator.simulate()
        **kwargs : dict, optional
            Additional keyword arguments to pass to the simulator.simulate()
        
        Returns
        -------
        Returns NState if the simulator didn't return anything, else returns the output of the simulator.

        Notes
        -----
        The simulator is forced to ONLY take NState (self) as an argument for initialization. 
        If the simulator requires additional arguments, initialize the simulator in an extra step, like so:
        ```python
        system.simulator = simulator(system, *args, **kwargs)
        system.simulator.simulate(*args, **kwargs)
        ```

        or
                
        ```python
        system.set_simulator(simulator, *args, **kwargs)
        system.simulator.simulate(*args, **kwargs)
        ```
        """
        if not simulator:
            if not self.simulator:
                self.log.error("Simulator not set. Use as an argument in NState.simulate() or set an initialized simulator to NState.simulator")
                return 
            else:
                simout = self.simulator.simulate(*args, **kwargs)
                if simout:
                    return simout
        else:
            self.log.info(f"Simulating with {simulator.__name__}.\n")
            simulator = simulator(self)
            simout = simulator.simulate(*args, **kwargs)
            if simout:
                return simout
        return self

    def set_simulator(self, simulator, *args, **kwargs) -> None:
        """
        Sets and initializes the simulator for the system. 

        Parameters
        ----------
        simulator : class
            The simulator class to use for the system. Unless using a custom simulator, 
            use the provided simulators in gekim.simulators.
        *args : tuple, optional
            Additional arguments to pass to the simulator for initialization.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the simulator for initialization.

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

        if self.simout is None:
            self.log.error("Simulated data not found in self.simout. Run a simulation first.")
            return None
        
        # simout can be a list or a np.ndarray depending on if initial concentrations were arrays or scalars
        if isinstance(self.simout["y"], list):
            total_y = [None]*len(self.simout["y"])
            simout_is_list = True
        elif isinstance(self.simout["y"], np.ndarray):
            total_y = np.zeros_like(self.simout["y"])
            simout_is_list = False
        else:
            self.log.error("Unrecognized simout data type. Expected list or np.ndarray.")
            return None

        for name in species_names:
            if name not in self.species:
                self.log.error(f"Species '{name}' not found in the system.")
                return None
            if self.species[name].simout is None:
                self.log.error(f"Simulated data not found for species '{name}'.")
                return None
            if simout_is_list:
                for i,simout in enumerate(self.species[name].simout["y"]):
                    if total_y[i] is None:
                        total_y[i] = simout
                    else:
                        total_y[i] += simout
            else:
                total_y += self.species[name].simout["y"]
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

    def find_paths(self, start_species: Union[str,Species], end_species: Union[str,Species], only_linear_paths=True, 
                   prob_cutoff=1e-10, max_depth=20, log_paths=False, normalize_prob=True):
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
            Cutoff probability to stop searching current path (default is 1e-10). This is before normalization of probabilities.
        max_depth : int, optional
            Maximum depth to limit the search (default is 20), ie max length of path - 1. 
        log_paths : bool, optional
            Whether to log the path strings found (default is False).

        Notes
        -------
        Saves a list of paths in `self.paths` sorted by probability.

        Probability may be misleading here due to the cutoffs and infinite possibilities of nonlinear paths. 
            
        Probability is calculated as the product of the transition probabilities, 
            which is the transition rate constant over the sum of available transition rate constants (markov chain-esque)
            
        """
        #TODO: use J_sym?
        #TODO: prob seems right, but why isnt it what is expected?
        #TODO: needs to be optimized, probably with multithreading. but since its main use is for finding linear systems, its fine
            #TODO: needs to be optimized in many ways, including algorithmic. Many wasted or repeat cycles  
        
        def get_transition_probability(transition, current_sp_name):
            k_sum = sum(tr.k for tr in self.transitions.values() if current_sp_name in [sp[0] for sp in tr.source])
            return transition.k / k_sum if k_sum > 0 else 0

        def dfs(current_sp_name, target_sp_name, visited_names, current_path, current_transitions, current_prob, depth):
            if current_prob < prob_cutoff or depth > max_depth:
                return
            
            if current_sp_name == target_sp_name:
                self.paths.append(Path(current_path[:], current_transitions[:], current_prob))
                return

            for transition in self.transitions.values():
                if current_sp_name in [sp[0] for sp in transition.source]:
                    next_species_list = [sp[0] for sp in transition.target]
                    if only_linear_paths and any(sp in visited_names for sp in next_species_list):
                        continue

                    for next_sp_name in next_species_list:
                        next_prob = current_prob * get_transition_probability(transition, current_sp_name)
                        #print(f"{current_sp_name} -> {next_sp_name} ({current_prob}->{next_prob}) by transition {transition.name}")
                        visited_names.add(next_sp_name)
                        current_path.append(self.species[next_sp_name])
                        current_transitions.append(transition)
                        dfs(next_sp_name, target_sp_name, visited_names, current_path, current_transitions, next_prob, depth + 1)
                        if only_linear_paths: #bandaid? did it work?
                            visited_names.remove(next_sp_name)
                        current_path.pop()
                        current_transitions.pop()

        # Input validation
        all_linear_tr = True
        for transition in self.transitions.values():
            if not transition.linear:
                all_linear_tr = False
                self.log.warning(f"Transition '{transition.name}' is not linear!")
        if not all_linear_tr:
            self.log.error("This method only uses TRANSITION.k to calculate probabilities, and expects single TRANSITION.source to contain only one species.\n" +
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

        # Normalize probabilities
        if normalize_prob:
            total_prob = sum(path.probability for path in self.paths)
            if total_prob > 0:
                for path in self.paths:
                    path.probability /= total_prob
    
        # Sort paths by probability
        self.paths.sort(key=lambda p: p.probability, reverse=True)

        
        if log_paths:
            self.log.info(f"\n{len(self.paths)} paths found from '{start_species.name}' to '{end_species.name}':")
            for path in self.paths:
                self.log.info(str(path))
        else:
            self.log.info(f"\n{len(self.paths)} paths found from '{start_species.name}' to '{end_species.name}'")
        
        return 

    def get_species_sets(self) -> dict:
        """
        Combine the probabilities of self.paths that contain the same set of species.
        Essentially a utility function if `NState.find_paths()` yields a ton of paths.

        Returns
        -------
        dict
            Dictionary with species sets as keys and combined probabilities as values.
        """
        paths = self.paths
        combined_paths = defaultdict(lambda: {"combined_probability": 0.0})

        for path in paths:
            species_set = frozenset(sp.name for sp in path.species_path)
            combined_paths[species_set]["combined_probability"] += path.probability

        # Sort combined paths by combined probability
        sorted_combined_paths = dict(sorted(combined_paths.items(), key=lambda item: item[1]['combined_probability'], reverse=True))

        self.log.info(f"\nSpecies sets and their combined probabilities (sorted):")
        for species_set, data in sorted_combined_paths.items():
            prob_fmt = "{:.2e}".format(data['combined_probability'])
            self.log.info(f"Combined P: {prob_fmt}, Species: {species_set}")

        return sorted_combined_paths
