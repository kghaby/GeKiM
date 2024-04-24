import numpy as np
import re
import copy
import logging
import sys
from scipy.integrate import solve_ivp
from itertools import product
        
class NState:
    #TODO: Add stochastic method
    #TODO: logger retains previous classes? Jupyter output was showing previous class logs i think. Happens sometimes
    
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
        # Document the order of the transitions
        for idx, name in enumerate(self.config['transitions']):
            self.transitions[name]['index'] = idx
        
        self._format_transitions()
        self._generate_matrices()
        # self._construct_ode_mat() # replace with jacobian and proper ode matrices

        #self.t = None # is it actually worth saving t?
        
        self.logger.info(f"NState system initialized successfully.")
    
    def reformat(self):
        """
        Use this if you added transitions or species after initialization.
        """
        self._format_transitions()
        self._generate_matrices()
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
                
    @staticmethod
    def _parse_species_string(species_str):
        """
        Extract coefficient and species name from species string.

        Parameters:
        species_str (str): A species string, e.g., '2A'.

        Returns:
        tuple: A tuple of species name (str) and stoichiometric coefficient (int).
        """
        match = re.match(r"(\d*\.?\d*)(\D.*)", species_str)
        if match and match.groups()[0]:
            coeff = float(match.groups()[0])
        else:
            coeff = 1
        name = match.groups()[1] if match else species_str
        return name,coeff
                
    def _format_transitions(self):
        """
        Format the transitions by extracting and combining coefficients and species names.
        Is idempotent.
        """
        for _, tr in self.transitions.items():
            for direction in ['from', 'to']:
                parsed_species = {}
                for sp in tr[direction]:
                    if isinstance(sp, str):
                        name, coeff = self._parse_species_string(sp)
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
                    if name in parsed_species:
                        parsed_species[name] += coeff # combine coeffs
                    else:
                        parsed_species[name] = coeff
                tr[direction] = [(name, coeff) for name, coeff in parsed_species.items()]
        return

    def log_dcdts(self,force_print=False):
        """
        Log the ordinary differential equations for the concentrations of each species over time in a readable format.
        Will write the coefficients used in the config.
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
            rate = f"{tr_name} * " + " * ".join([f"{sp_name}^{coeff}" if coeff != 1 else f"{sp_name}" for sp_name,coeff in tr['from']])
            rate = rate.rstrip(" *")  # Remove trailing " *"

            # Add rate law to the eqns
            for sp_name,coeff in tr['from']:
                term = f"{coeff} * {rate}" if coeff > 1 else rate
                dcdt_dict[sp_name].append(f" - {term}")

            for sp_name,coeff in tr['to']:
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
        return

    def _generate_matrices(self):
        """
        Generates 
            unit species matrix (self._unit_sp_mat), 
            stoichiometry matrix (self._stoich_mat), 
            stoichiometry reactant matrix (self._stoich_reactant_mat), and 
            rate constant vector (self._k_vec) and diagonal matrix (self._k_diag).

        Rows are transitions, columns are species.

        Used for solving ODEs.
        """

        n_species = len(self.species)
        n_transitions = len(self.transitions)
        self._unit_sp_mat = np.eye(n_species, dtype=int)

        # Initialize matrices
        self._stoich_reactant_mat = np.zeros((n_transitions, n_species))
        self._stoich_mat = np.zeros((n_transitions, n_species))
        self._k_vec = np.zeros(n_transitions)

        # Fill stoich matrices and k vector
        for tr_name, tr in self.transitions.items():
            tr_idx = tr['index']
            self._k_vec[tr_idx] = tr['value']
            reactant_vec = np.sum([self._unit_sp_mat[self.species[name]['index']] * coeff for name, coeff in tr['from']],axis=0)
            product_vec = np.sum([self._unit_sp_mat[self.species[name]['index']] * coeff for name, coeff in tr['to']],axis=0)
            
            self._stoich_reactant_mat[tr_idx, :] = reactant_vec  
            #self._stoich_product_mat[tr_idx, :] = product_vec   # not used
            self._stoich_mat[tr_idx] = product_vec - reactant_vec

        self._k_diag = np.diag(self._k_vec)

        return
    
    def _dcdt(self, t, conc):
        #TODO: Use higher dimensionality conc arrays to process multiple input concs at once
        C_Nr = np.prod(np.power(conc, self._stoich_reactant_mat), axis=1) # state dependencies
        N_K = np.dot(self._k_diag,self._stoich_mat) # interactions
        dCdt = np.dot(C_Nr,N_K)
        return dCdt

    def solve_odes(self, t, conc0_dict=None, method='BDF', rtol=1e-6, atol=1e-8, output_raw=False):
        """
        Solve the ODEs for the system and update the species concentrations.

        Parameters:
        t (np.array): Time points for ODE solutions.

        conc0_dict (dict: {str:np.array}): Dictionary of {species_name: conc0_arr} pairs for initial concentrations to simulate. 
            Unprovided species will use self.species['name']['conc'][0] as a single-point initial concentration.
            Using multiple conc0's will nest the concentrations in an array and raw solutions in a list.
            Default is None, ie all initial concentrations are single point from the self.species dict.
            Example: {"Ligand":np.linspace(1,1500,100)} for a Michaelis-Menten ligand concentration scan.
        method (str): Integration method, default is 'BDF'.
        rtol (float): Relative tolerance for the solver. Default is 1e-6
        atol (float): Absolute tolerance for the solver. Default is 1e-8
        output_raw (bool): If True, return raw solver output. 

        """
        #TODO: add option to simulate to convergence
            # would help with low values of ligand and conc0 arrays where both ends of the spectrum need very different endpoints
            # rougly predict how much time is needed. maybe by plugging
        self.log_dcdts()
        
        if conc0_dict:
            combinations = product(*(
                np.atleast_1d(conc0_dict.get(sp_name, [np.atleast_1d(sp_data['conc']).flatten()[0]])) 
                for sp_name, sp_data in self.species.items()
            ))
            conc0_mat = np.vstack([comb for comb in combinations])
        else:
            conc0_mat = np.atleast_2d([np.atleast_1d(sp_data['conc']).flatten()[0] for _, sp_data in self.species.items()])
        conc0_iterations_amt = conc0_mat.shape[0]

        solutions = []
        for conc0 in conc0_mat:
            solution = solve_ivp(self._dcdt, (t[0], t[-1]), conc0, method=method, t_eval=t, rtol=rtol, atol=atol) # vectorized=True makes it slower I think bc low len(conc0)
            if not solution.success:
                raise RuntimeError("ODE solver failed: " + solution.message)
            solutions.append(solution)
            
        self.logger.info("ODEs solved successfully. Saving data...")

        for name, data in self.species.items():
            conc_shape = (conc0_iterations_amt,len(t))
            if conc0_iterations_amt == 1:
                data['conc'] = solution.y[data['index']]
            else:
                data['conc'] = np.zeros(conc_shape)
                for i, solution in enumerate(solutions):
                    data['conc'][i] = solution.y[data['index']]

        self.logger.info(f"{conc0_iterations_amt} concentration vectors of {name} saved to self.species['{name}']['conc']")

        if output_raw:
            if conc0_iterations_amt == 1:
                self.logger.info("Returning raw solver output.")
                return solutions[0]
            self.logger.info("Returning list of raw solver outputs.")        
            return solutions
        else:
            self.logger.info("Not returning raw solver output. Use output_raw=True to return raw data.")
            return
        
    def simulate(self, t, output_raw=False):
        """
        Simulate the system stochastically using the Gillespie algorithm.

        Parameters:
        t (np.array): Time points for desired observations.
        output_raw (bool): If True, return raw simulation data.

        Returns:
        Dict or None, depending on output_mode.
        """
        #TODO: Although i do like the idea of being able to continue from a previous run, so add an option called "continue" which takes an integer which points to the index of the run that its continuing from
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
                tr_rate = tr['value'] * np.prod([conc[self.species[sp_name]['index']] ** coeff for sp_name,coeff in tr['from']])
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




