import numpy as np
import re
import copy
import logging
import sys
from scipy.integrate import solve_ivp
from numba import njit
from ..fortran.kinetics import kinetics

@njit
def _dcdt(t, concentrations, rate_constants, stoich_matrix_from, stoich_matrix_to,num_reactions):
    """
    Compute the derivative of concentrations with respect to time.

    Parameters:
    t (float): Time.
    concentrations (numpy.array): Array of species concentrations.

    Returns:
    numpy.array: Array of concentration time derivatives.
    """
    dcdt_arr = np.zeros_like(concentrations)
    for i in range(num_reactions):
        rate = rate_constants[i]
        reactant_mask = stoich_matrix_from[:, i] > 0
        for j, is_reactant in enumerate(reactant_mask):
            if is_reactant:
                rate *= concentrations[j] ** stoich_matrix_from[j, i]
        net_change = stoich_matrix_to[:, i] - stoich_matrix_from[:, i]
        dcdt_arr += rate * net_change
    return dcdt_arr
        
class NState:
    #TODO: Make sure its all np arrays and not lists 
    #TODO: Add stochastic method
    
    def __init__(self, config, logfilename=None, quiet=False):
        """
        Initialize the NState class with configuration data.

        Parameters:
        config (dict): Configuration containing species and transitions.
                       Species should contain name, initial concentration, and label.
                       Transitions should contain name, from-species, to-species, value, and label.

        Raises:
        ValueError: If config is invalid.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.handlers = []
        if quiet:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.INFO)

        if logfilename:
            file_handler = logging.FileHandler(logfilename)
            #file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        #stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stream_handler)

        self._validate_config(config)
        
        self.config = copy.deepcopy(config)
    
        self.species = self.config['species']
        # Document the order of the species
        for idx, name in enumerate(self.config['species']):
            self.species[name]['index'] = idx
        self._validate_species()

        self.transitions = self.config['transitions']
        self._preprocess_transitions()

        self.t = None
        self._rate_constants = np.array([tr['value'] for tr in self.transitions.values()])
        self._stoich_matrix_from, self._stoich_matrix_to = self._build_stoichiometry_matrix()
        
        self.logger.info(f"NState system initialized successfully.")

    def _validate_config(self,config):
        #TODO: Make sure names are unique and nonempty 
        if not 'species' in config or not 'transitions' in config:
            raise ValueError("Config must contain 'species' and 'transitions' keys.")

    def _validate_species(self):
        """
        Validate the species data in the configuration.

        Returns:
        bool: True if valid, False otherwise.
        """
        labels = set()
        for name, data in self.species.items():
            # Validate labels
            label = data.get('label',name)
            if label in labels:
                self.logger.error(f"Duplicate label '{label}' found for species '{name}'.")
                return False
            labels.add(label)
        
            # Validate concentrations
            if not 'conc' in data.keys():
                raise ValueError("Initial concentration not found in species[name]['conc'].")
            conc = data.get('conc')
            if not isinstance(conc, np.ndarray):
                data['conc'] = np.array([conc])
        return True
                
    def sum_conc(self,whitelist:list=None,blacklist:list=None):
        """
        Sum the concentrations of specified species in the model.
        Whitelist and blacklist cannot be provided simultaneously.

        Parameters:
        whitelist (list, optional): Names of species to include in the sum.
        blacklist (list, optional): Names of species to exclude from the sum.

        Returns:
        float: The sum of the concentrations.
        """
        if whitelist and blacklist:
            raise ValueError("Provide either a whitelist or a blacklist, not both.")

        species_names = self.species.keys()
        
        if whitelist:
            species_names = [name for name in whitelist if name in species_names]
        elif blacklist:
            species_names = [name for name in species_names if name not in blacklist]

        total_concentration = np.sum([self.species[name]['conc'] for name in species_names], axis=0)

        return total_concentration


    def _preprocess_transitions(self):
        """
        Preprocess the transitions by extracting coefficients and species names.
        """
        for _, tr in self.transitions.items():
            tr['from'] = [self._identify_coeff(s) for s in tr['from']]
            tr['to'] = [self._identify_coeff(s) for s in tr['to']]

    @staticmethod
    def _identify_coeff(species_str):
        """
        Extract coefficient and species name from species string.

        Parameters:
        species_str (str): A species string, e.g., '2A'.

        Returns:
        tuple: A tuple of coefficient (int) and species name (str).
        """
        match = re.match(r"(\d*)(\D.*)", species_str)
        coeff = int(match.groups()[0]) if match and match.groups()[0] else 1
        name = match.groups()[1] if match else species_str
        return coeff, name
    
    def _build_stoichiometry_matrix(self):
        num_species = len(self.species)
        num_transitions = len(self.transitions)
        stoich_matrix_from = np.zeros((num_species, num_transitions))
        stoich_matrix_to = np.zeros((num_species, num_transitions))
        
        for tr_idx, (tr_name, tr) in enumerate(self.transitions.items()):
            for reactant in tr['from']:
                coeff, name = reactant
                sp_idx = self.species[name]['index']
                stoich_matrix_from[sp_idx, tr_idx] = coeff
            
            for product in tr['to']:
                coeff, name = product
                sp_idx = self.species[name]['index']
                stoich_matrix_to[sp_idx, tr_idx] = coeff
        #net_stoich_matrix = (stoich_matrix_from - stoich_matrix_to)

        return stoich_matrix_from,stoich_matrix_to

    def _dcdt(self, t, concentrations):
        """
        Slightly slower than vectorized algorithm due to more rigorous iteration.
        Compute the derivative of concentrations with respect to time.

        Parameters:
        t (float): Time.
        concentrations (numpy.array): Array of species concentrations.

        Returns:
        numpy.array: Array of concentration time derivatives.
        """
        #TODO: vectorize and use numba by preprocessing to cover edge cases. Also, write tests for edge cases, esp regarding stoichiometry. I might have the test configs somewhere 
        dcdt_arr = np.zeros_like(concentrations)
        for tr in self.transitions.values():
            rate_constant = tr['value']
            rate = rate_constant * np.prod([concentrations[self.species[sp_name]['index']] ** coeff for coeff, sp_name in tr['from']])
            # Iterating through like this is beneficial because it captures stoichiometry that is evident in the list rather than coefficient 
                # (eg "from": ["E","E", "I"] is equal to "from": ["2E", "I"])
            for coeff, sp_name in tr['from']:
                dcdt_arr[self.species[sp_name]['index']] -= coeff * rate
            for coeff, sp_name in tr['to']:
                dcdt_arr[self.species[sp_name]['index']] += coeff * rate
        return dcdt_arr
        
    def log_dcdts(self,force_print=False):
        """
        Log the ordinary differential equations for the concentrations of each species over time in a readable format.
        """
        dcdt_dict = {}
        max_header_length = 0

        # Find the max length for the headers
        for sp_name in self.species:
            header_length = len(f"d[{sp_name}]/dt")
            max_header_length = max(max_header_length, header_length)

        # Write eqn headers and rate laws
        for sp_name in self.species:
            dcdt_dict[sp_name] = [f"d[{sp_name}]/dt".ljust(max_header_length) + " ="]

        for tr_name, tr in self.transitions.items():
            # Write rate law
            rate = f"{tr_name} * " + " * ".join([f"{sp_name}^{coeff}" if coeff > 1 else f"{sp_name}" for coeff, sp_name in tr['from']])
            rate = rate.rstrip(" *")  # Remove trailing " *"

            # Add rate law to the eqns
            for coeff, sp_name in tr['from']:
                term = f"{coeff} * {rate}" if coeff > 1 else rate
                dcdt_dict[sp_name].append(f" - {term}")

            for coeff, sp_name in tr['to']:
                term = f"{coeff} * {rate}" if coeff > 1 else rate
                dcdt_dict[sp_name].append(f" + {term}")

        # Construct the final string
        ode_log = "ODEs:\n\n"
        for sp_name, eqn_parts in dcdt_dict.items():
            # Aligning '+' and '-' symbols
            eqn_header = eqn_parts[0]
            terms = eqn_parts[1:]
            aligned_terms = [eqn_header + " " + terms[0]] if terms else [eqn_header]
            aligned_terms += [f"{'':>{max_header_length + 3}}{term}" for term in terms[1:]]
            formatted_eqn = "\n".join(aligned_terms)
            ode_log += formatted_eqn + '\n\n'

        self.logger.info(ode_log)
        if force_print:
            print(ode_log)
    


    def simulate_deterministic(self, t, method='BDF', rtol=1e-6, atol=1e-8, output_raw=False):
        """
        Solve the ODEs for the system with flexible output handling.

        Parameters:
        t (np.array): Time points for ODE solutions.
        method (str): Integration method, default is 'BDF'.

        rtol (float): Relative tolerance for the solver. Default is 1e-6
        atol (float): Absolute tolerance for the solver. Default is 1e-8
        output_raw (bool): If True, return raw solver output. 

        Returns:
        Dict or None, depending on output_mode.
        """
        conc0 = np.array([np.atleast_1d(sp['conc'])[0] for _, sp in self.species.items()])
        t_span = (t[0], t[-1])
        self.log_dcdts()
        # num_reactions = self._rate_constants.shape[0] #used in faster alg
        try:
            solution = solve_ivp(
                fun=lambda t, conc: self._dcdt(t, conc),
                t_span=t_span, y0=conc0, t_eval=t, method=method, rtol=rtol, atol=atol
            )
            if not solution.success:
                raise RuntimeError("ODE solver failed: " + solution.message)

            self.t = t
            for name,data in self.species.items():
                data['conc'] = solution.y[data['index']]

            self.logger.info("ODEs solved successfully.")

            if output_raw:
                return solution
            else:
                return
        except Exception as e:
            self.logger.error(f"Error in solving ODEs: {e}")
            raise
    
    def _simulate_deterministic(self, t, method='BDF', rtol=1e-6, atol=1e-8, output_raw=False):
        """
        Uses fortran to calc ODE set. A tad slower.
        Solve the ODEs for the system with flexible output handling.

        Parameters:
        t (np.array): Time points for ODE solutions.
        method (str): Integration method, default is 'BDF'.

        rtol (float): Relative tolerance for the solver. Default is 1e-6
        atol (float): Absolute tolerance for the solver. Default is 1e-8
        output_raw (bool): If True, return raw solver output. 

        Returns:
        Dict or None, depending on output_mode.
        """
        def odes(t, conc):
            dcdt = np.zeros_like(conc)  # dcdt also needs to be of dtype=np.float32 to match Fortran expectations
            kinetics.calc_dcdt(conc, self._rate_constants, self._stoich_matrix_from, self._stoich_matrix_to, dcdt, num_species)
            return dcdt
        conc0 = np.array([np.atleast_1d(sp['conc'])[0] for _, sp in self.species.items()])
        t_span = (t[0], t[-1])
        self.log_dcdts()
        num_species = len(self.species)
        try:
            solution = solve_ivp(
                fun=odes,
                t_span=t_span, y0=conc0, t_eval=t, method=method, rtol=rtol, atol=atol
            )
            if not solution.success:
                raise RuntimeError("ODE solver failed: " + solution.message)

            self.t = t
            for name,data in self.species.items():
                data['conc'] = solution.y[data['index']]

            self.logger.info("ODEs solved successfully.")

            if output_raw:
                return solution
            else:
                return
        except Exception as e:
            self.logger.error(f"Error in solving ODEs: {e}")
            raise

    def simulate_stochastic(self, t, output_raw=False):
        """
        Simulate the system stochastically using the Gillespie algorithm.

        Parameters:
        t (np.array): Time points for desired observations.
        output_raw (bool): If True, return raw simulation data.

        Returns:
        Dict or None, depending on output_mode.
        """
        # Initialize
        current_time = 0.0
        conc = np.array([np.atleast_1d(sp['conc'])[0] for _, sp in self.species.items()])
        times = [current_time]
        concentrations = [conc.copy()]
        transitions_list = list(self.transitions.values())  # Convert dictionary values to a list for indexing

        # Simulation loop
        while current_time < t[-1]:
            rate = np.zeros(len(transitions_list))
            for tr_idx, tr in enumerate(transitions_list):
                tr_rate = tr['value'] * np.prod([conc[self.species[sp_name]['index']] ** coeff for coeff, sp_name in tr['from']])
                rate[tr_idx] = tr_rate

            total_rate = np.sum(rate)
            if total_rate == 0:
                break

            # Time to next event
            tau = np.random.exponential(1/total_rate)
            current_time += tau
            if current_time > t[-1]:
                break

            # Determine which event occurs
            cumulative_rate = np.cumsum(rate)
            event = np.searchsorted(cumulative_rate, np.random.rand() * total_rate)

            # Update concentrations
            for coeff, sp_name in transitions_list[event]['from']:
                conc[self.species[sp_name]['index']] -= coeff
            for coeff, sp_name in transitions_list[event]['to']:
                conc[self.species[sp_name]['index']] += coeff

            times.append(current_time)
            concentrations.append(conc.copy())

        # Interpolate or sample to requested time points
        interpolated_concs = np.zeros((len(self.species), len(t)))
        j = 0
        for i, desired_time in enumerate(t):
            while j < len(times) - 1 and times[j+1] < desired_time:
                j += 1
            interpolated_concs[:, i] = concentrations[j]

        # Logging
        self.logger.info("Stochastic simulation completed successfully.")

        if output_raw:
            return {'time': np.array(times), 'concentrations': np.array(concentrations)}
        else:
            # Format output to match deterministic function if needed
            solution = {'time': t, 'concentrations': interpolated_concs}
            for idx, sp in enumerate(self.species.keys()):
                self.species[sp]['conc'] = solution['concentrations'][idx]
            return solution





