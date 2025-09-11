import numpy as np
from collections import defaultdict
from sympy import Symbol
from typing import Optional, Union

from ..utils.logging import Logger
from ..schemes.scheme import Scheme
from ..schemes.species import Species
from ..schemes.transition import Transition
from .path import Path

class System:
    """
    Core System class that binds a scheme (model structure) with a Simulator for execution.
    Provides methods to simulate and analyze the system.
    """
    def __init__(self, scheme: Union[dict, Scheme] = None, logfilename: Optional[str] = None, quiet: bool = False):
        """
        Initialize the System class with configuration data. Can be any degree of nonlinearity.

        Parameters
        ----------
        scheme : dict
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
        self._quiet = quiet # set quiet without triggering setter
        
        self.scheme = scheme # invoke setter to validate and set scheme
        
        self.simulator = None
        self.log.info("System initialized successfully.")
        self.log.info("="*80)

    @property
    def quiet(self) -> bool:
        return self._quiet

    @quiet.setter
    def quiet(self, value: bool):
        self._quiet = value
        self.log.quiet = value
        
    @property
    def scheme(self) -> Scheme:
        return self._scheme
    
    @scheme.setter
    def scheme(self, config: Union[dict, Scheme, None]):
        if config is None:
            self._scheme = Scheme(log=self.log)
            self.log.info("Initialized empty Scheme.\n")
        elif isinstance(config, dict):
            self._scheme = Scheme(config, log=self.log)
            self.log.info("Initialized Scheme from dict.\n")
        elif isinstance(config, Scheme):
            self._scheme = config
            self.log.info("Scheme object already initialized.")
            self.log.info("\tAssigning System logger to Scheme.")
            self._scheme.log = self.log
            
        else:
            raise TypeError("scheme must be None, dict, or Scheme")
        
        # side‑effects on swap:
        self._reset_simulator()

    @property
    def species(self) -> dict[str, Species]:
        return self.scheme.species

    @property
    def transitions(self) -> dict[str, Transition]:
        return self.scheme.transitions
    
    @property
    def simin(self):
        if self.simulator and self.simulator.simin:
            return self.simulator.simin
        else:
            self.log.error("Simulator not set or simin not initialized. "
                           "Use System.set_simulator() to set a simulator.")
        return {}

    @property
    def simout(self):
        if self.simulator and self.simulator.simout:
            return self.simulator.simout
        else:
            self.log.error("Simulator not set or simout not initialized. "
                           "Use System.set_simulator() to set a simulator.")
        return {}

    def set_scheme(self, scheme: Union[dict, Scheme], name: Optional[str] = None,
                   color_kwargs: Optional[dict] = None) -> Scheme:
        """
        Sets the scheme for the system. 

        Parameters
        ----------
        scheme : dict or Scheme
        
        name : str, optional
        
        color_kwargs : dict, optional
            Keyword arguments for color assignment to species
            See `gekim.utils.plotting.assign_colors_to_species()` for details.

        Returns
        -------
        The scheme instance (which is now an attribute of the system).
        """
        if isinstance(scheme, dict):
            scheme = Scheme(scheme, name=name, color_kwargs=color_kwargs)
            self.log.info("Initialized Scheme from dict.")
        self.scheme = scheme
        self.log.info("Scheme set successfully.")
        self.log.info(self.scheme)
        self.log.info("="*80)
        return self.scheme

    def set_simulator(self, simulator_class, *args, **kwargs):
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
        
        Returns
        -------
        The simulator instance (which is now an attribute of the system).

        Notes
        ----- 
        Alternatively, you can set the simulator directly as an attribute of the system:
        ```python
        system.simulator = simulator(system)
        system.simulator.simulate(...)
        ```
        This method may be more IDE-friendly.
        """
        self.simulator = simulator_class(self, *args, **kwargs)
        self.log.info(f"Simulator set to {simulator_class.__name__}.")
        self.log.info("Use System.simulate() to run the simulation.")
        self.log.info("="*80)
        return self.simulator
    
    def _reset_simulator(self):
        """
        Reset the simulator and simulation output.
        This is called when the scheme is changed.
        
        Notes
        -----
        This method does not reinit the simulator (in case arguments are needed)!
        """
        self.simulator = None
        self.log.info("Simulator and simulation output reset.\n")
        

    def simulate(self, simulator_class=None, *args, **kwargs):
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
        Returns System if the simulator didn't return anything, else returns the output of the simulator.

        Notes
        -----
        The simulator is forced to ONLY take System (self) as an argument for initialization. 
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
        if not simulator_class:
            if not self.simulator:
                self.log.error("Simulator not set. Use as an argument in System.simulate() or set an initialized simulator to System.simulator")
                return
            else:
                simout = self.simulator.simulate(*args, **kwargs)
                if simout:
                    return simout
        else:
            self.log.info(f"Simulating with {simulator_class.__name__}.\n")
            simulator_class = simulator_class(self)
            simout = simulator_class.simulate(*args, **kwargs)
            if simout:
                return simout
        return self

    def set_parameter(self, param: Union[str, Symbol], value: float):
        """
        Assign a numeric value to a model parameter (rate constant or symbolic initial concentration).
        `param` can be the parameter name or the sympy Symbol object.
        """
        if isinstance(param, str):
            if param in self.scheme.param_symbols:
                sym = self.scheme.param_symbols[param]
            else:
                raise KeyError(f"Parameter '{param}' not found in scheme.")
        else:
            sym = param
            # Verify symbol is recognized
            found = False
            for name, sym_obj in self.scheme.param_symbols.items():
                if sym_obj == sym:
                    found = True
                    break
            if not found:
                raise KeyError(f"Parameter symbol '{sym}' is not part of the scheme.")
        # Set numeric value
        self.scheme.param_values[sym] = value

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

        if isinstance(whitelist, str):
            whitelist = [whitelist]
        if isinstance(blacklist, str):
            blacklist = [blacklist]
            
        if whitelist:
            species_names = [name for name in whitelist if name in species_names]
        elif blacklist:
            species_names = [name for name in species_names if name not in blacklist]

        if self.simout is None:
            self.log.error("Simulated data not found in self.simout. Run a simulation first.")
            return None
        # simout can be a list or a np.ndarray depending on if initial concentrations were arrays or scalars
        if isinstance(self.simout["y"], list):
            len_simouts = len(self.simout["y"])
            total_y = [np.zeros_like(self.simout["y"][i][0]) for i in range(len_simouts)]
            simout_is_list = True
        elif isinstance(self.simout["y"], np.ndarray):
            first_species_simout = self.simout["y"][0]
            total_y = np.zeros_like(first_species_simout) 
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
                simouts = self.species[name].simout["y"]
                for i,simout in enumerate(simouts):
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
    
    def find_paths(self, start_species: Union[str,Species], end_species: Union[str,Species], 
                   only_linear_paths=True, prob_cutoff=1e-10, max_depth=20, 
                   log_paths=False, normalize_prob=True):
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
            self.log.error("This method only uses TRANSITION.k to calculate probabilities, and expects TRANSITION.source to contain only one species.\n" +
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
        Essentially a utility function if `System.find_paths()` yields a ton of paths.

        Returns
        -------
        dict
            Dictionary with species sets as keys and combined probabilities as values.
        """
        if not hasattr(self, 'paths') or not self.paths:
            self.log.error("No paths found. Run System.find_paths() first.")
            return {}
        paths = self.paths
        combined_paths = defaultdict(lambda: {"combined_probability": 0.0})

        for path in paths:
            species_set = frozenset(sp.name for sp in path.species_path)
            combined_paths[species_set]["combined_probability"] += path.probability

        # Sort combined paths by combined probability
        sorted_combined_paths = dict(sorted(combined_paths.items(), key=lambda item: item[1]['combined_probability'], reverse=True))

        self.log.info("\nSpecies sets and their combined probabilities (sorted):")
        for species_set, data in sorted_combined_paths.items():
            prob_fmt = "{:.2e}".format(data['combined_probability'])
            self.log.info(f"Combined P: {prob_fmt}, Species: {species_set}")

        return sorted_combined_paths
