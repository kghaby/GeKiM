import numpy as np
import re
import copy
import logging
import sys
from scipy.integrate import solve_ivp
from itertools import product
from sympy import symbols, Matrix, prod, pretty, zeros, lambdify
from ..utils import integerable_float
        
class NState:
    #TODO: Add stochastic method
    #TODO: use sympy for odes so that massive scheme odes are easily dictionaried and utilized

    
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
        self._generate_matrices() # generates self._unit_sp_mat, self._stoich_mat, self._stoich_reactant_mat, self._k_vec, self._k_diag

        self.log_dcdts()
        
        self._generate_jac() # generates self.J_sym, self.J_symsp_numtr, self.J_func_wrap
        self.log_jac()

        self.t_dcdts = None # is it actually worth saving t?
        
        self.logger.info(f"NState system initialized successfully.\n")
    
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
        #TODO: use assign color 
        labels = set()
        for name, data in self.species.items():
            # Validate labels
            label = data.get('label',name)
            if 'label' not in data.keys():
                self.logger.info(f"Label not found for species '{name}'. Using species name as label.")
                self.species[name]['label'] = name
            if label in labels:
                self.logger.error(f"Duplicate label '{label}' found for species '{name}'.")
                return False
            labels.add(label)
        
            # Validate concentrations
            if not 'conc' in data.keys():
                raise ValueError(f"Initial concentration not found in species['{name}']['conc'].")
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
            coeff = integerable_float(coeff)
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
        Log the ordinary differential equations for the concentrations of each species over time.
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

    def _generate_jac(self):
        """
        Generate the symbolic Jacobian matrix and convert it to a numerical function.
        Saves the symbolic Jacobian to self.J_sym and the (wrapped) numerical function to self.J_func_wrap(t,y).
        Also saves a Jacobian with symbolic species and numeric transition rate constants to self.J_symsp_numtr.

        The wrapped numerical function is time-independent even though it accepts t as an argument!

        The Jacobian matrix here represents the first-order partial derivatives of the rate of change equations
        for each species with respect to all other species in the system.
        """

        sp_syms = {name: symbols(name) for name in self.species}
        tr_syms = {name: symbols(name) for name in self.transitions}
        tr_values = {name: self.transitions[name]['value'] for name in self.transitions}
        n_species = len(self.species)
        
        def make_dcdt_vec(tr_dict):
            # Rate laws using transition dictionary 
            rate_laws = {
                tr_name: tr_dict[tr_name] * prod(sp_syms[sp_name]**coeff for sp_name, coeff in tr['from'])
                for tr_name, tr in self.transitions.items()
            }
            # Construct the ODEs
            dcdt_vec = Matrix([0] * len(sp_syms))
            for tr_name, tr in self.transitions.items():
                for sp_name, coeff in tr['from']:
                    dcdt_vec[self.species[sp_name]['index']] -= coeff * rate_laws[tr_name]
                for sp_name, coeff in tr['to']:
                    dcdt_vec[self.species[sp_name]['index']] += coeff * rate_laws[tr_name]
            return dcdt_vec
    

        # Rate laws using symbolic transition names for readability in addition to symbolic species names
        dcdt_vec_sym = make_dcdt_vec(tr_syms)
        
        # Rate laws for actual computation with only symbolic species names
        dcdt_vec_num = make_dcdt_vec(tr_values)
        
        species_vec = Matrix(list(sp_syms.values()))

        # Symbolic Jacobian
        self.J_sym = dcdt_vec_sym.jacobian(species_vec)

        # Numerical Jacobian
        self.J_symsp_numtr = dcdt_vec_num.jacobian(species_vec) # Symbolic species, numeric transition rate constants
        J_func = lambdify(species_vec, self.J_symsp_numtr, 'numpy') # Make numerical function. Accepts 
        self.J_func_wrap = lambda t, y: J_func(*y) # Wrap J_func so that t,y is passed to the function to be compatible with solve_ivp

        return
      
    def log_jac(self):
        """
        Log the symbolic representation of the Jacobian matrix.
        The logged Jacobian includes row and column labels, but self.J_sym does not.

        The Jacobian matrix here represents the first-order partial derivatives of the rate of change equations
        for each species with respect to all other species in the system.
        """
        n_species = len(self.species)
        J_log = zeros(n_species + 1, n_species + 1)
        J_log[1:, 1:] = self.J_sym

        for sp_name,sp_data in self.species.items():
            J_log[0, sp_data['index']+1] = symbols(sp_name)
            J_log[sp_data['index']+1, 0] = symbols(f'd[{sp_name}]/dt')
        J_log[0,0] = symbols("_")

        J_log_str = "Jacobian (including row and column labels):\n"
        J_log_str += pretty((J_log),use_unicode=True)
        J_log_str += "\n"
        self.logger.info(J_log_str)
        
        return

    def _dcdt(self, t, conc):
        """
        Cannot model rates that are not simple power laws (eg dynamic inhibition, cooperativity, time dependent params). 
        But most of these can be baked in on the schematic level I think. 
        """
        #TODO: Use higher dimensionality conc arrays to process multiple input concs at once? Hard
        C_Nr = np.prod(np.power(conc, self._stoich_reactant_mat), axis=1) # state dependencies
        N_K = np.dot(self._k_diag,self._stoich_mat) # interactions
        dCdt = np.dot(C_Nr,N_K)
        return dCdt

    def solve_dcdts(self, t_eval=None, t_span=None, conc0_dict=None, method='BDF', rtol=1e-6, atol=1e-8, 
                    output_raw=False, dense_output=False):
        """
        Solve the ODEs of species concentration wrt time for the system. 
        Will update self.species['conc'] with the respective solutions.

        Arguments:
        t_eval (np.array): Time points for ODE solutions.

        t_span (tuple): Time span for ODE solutions.

        conc0_dict (dict: {str:np.array}): Dictionary of {species_name: conc0_arr} pairs for initial concentrations to simulate. 
            Unprovided species will use self.species[name]['conc'][0] as a single-point initial concentration.
            Using multiple conc0's will nest the concentrations in an array and raw solutions in a list.
                The conc0 combinations are saved to self.conc0_mat. 
                If not using conc0_dict, self.conc0_mat will still be set to the single conc0 vector. 
            Default is None, ie all initial concentrations are single point from the self.species dict.
            Example: {"Ligand":np.linspace(1,1500,100)} for a Michaelis-Menten ligand concentration scan.
        
        method (str): Integration method, default is 'BDF'.

        rtol (float): Relative tolerance for the solver. Default is 1e-6
            
        atol (float): Absolute tolerance for the solver. Default is 1e-8

        output_raw (bool): If True, return raw solver output.

        dense_output (bool): If True, save a scipy.integrate.OdeSolution instance to self.soln_continuous(t)
            If using multiple conc0's, this will be a list of functions that share indexing with the other outputs,
                and can be called like "self.soln_continuous[idx](t)".
            Access a specific species conc like soln_continuous(t)[self.species[name]['index']].

        """
        #TODO: check how combinations are arranged and make sure its intuitive to separate and use them (ie the indexing is clear)
        #TODO: More analytically approximate the time scale.
            # incorporate the network of transitions, nonlinearity, etc?
            # Linear scaling of the inverse min eigenvalue underestimates when conc0E ~= conc0I
            # Linearize system then use normal mode frequency of linear system (1/(sqrt(smallest eigenvalue))?
            # needs to be an n-dimensional function, where n is the degree of (non)linearity 
        if conc0_dict:
            combinations = product(*(
                np.atleast_1d(conc0_dict.get(sp_name, [np.atleast_1d(sp_data['conc']).flatten()[0]])) 
                for sp_name, sp_data in self.species.items()
            ))
            conc0_mat = np.vstack([comb for comb in combinations])
        else:
            conc0_mat = np.atleast_2d([np.atleast_1d(sp_data['conc']).flatten()[0] for _, sp_data in self.species.items()])
        conc0_mat_len = len(conc0_mat)
        if conc0_mat_len != 1:
            self.logger.info(f"Simulating {conc0_mat_len} initial concentration vectors...")
        self.conc0_mat = conc0_mat

        solns = []
        for conc0 in conc0_mat:
            if t_span is None:
                if t_eval is None:
                    # Estimate the timespan needed for convergence based on the smallest magnitude of the Jacobian eigenvalues at initial conditions
                    eigenvalues = np.linalg.eigvals(self.J_func_wrap(None, conc0))
                    eigenvalue_threshold = 1e-6 # below 1e-6 is considered insignificant. float32 cutoff maybe
                    filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) > eigenvalue_threshold] 
                    if filtered_eigenvalues.size == 0:
                        raise ValueError("No eigenvalues above the threshold, unable to estimate time scale.")
                    naive_time_scale = 1 / (np.abs(filtered_eigenvalues).min())
                    naive_time_scale = naive_time_scale * 6.5
                    t_span = (0, naive_time_scale) # Start at 0 or np.abs(filtered_eigenvalues).min()?
                    #print(f"Estimated time scale: {naive_time_scale:.2e}")
                    
                else:
                    t_span = (t_eval[0], t_eval[-1])

            soln = solve_ivp(self._dcdt, t_span=t_span, y0=conc0, method=method, t_eval=t_eval, 
                                rtol=rtol, atol=atol, jac=self.J_func_wrap, dense_output=dense_output) 
                # vectorized=True makes legacy dcdt func slower bc low len(conc0) I think
            if not soln.success:
                raise RuntimeError("FAILED: " + soln.message)
            solns.append(soln)
            
        self.logger.info("ODEs solved successfully. Saving data...")

        if conc0_mat_len == 1:
            self.t_dcdts = soln.t
            self.logger.info(f"\tTime saved to self.t_dcdts (np.array)")
            for _, data in self.species.items():
                data['conc'] = soln.y[data['index']]
            self.logger.info(f"\tConcentrations saved respectively to self.species[sp_name]['conc'] (np.array)")
            if dense_output:
                self.soln_continuous = soln.sol
                self.logger.info(f"\tSaving continuous solution function to self.soln_continuous(t) (scipy.integrate.OdeSolution)")
            else:
                self.soln_continuous = None
                self.logger.info("\tNot saving continuous solution. Use dense_output=True to save it to self.soln_continuous")
        else:
            self.t_dcdts = [soln.t for soln in solns] 
            self.logger.info(f"\t{conc0_mat_len} time vectors saved to self.t_dcdts (list of np.arrays)")
            for _, data in self.species.items():
                data['conc'] = [soln.y[data['index']] for soln in solns]
            self.logger.info(f"\t{conc0_mat_len} concentration vectors saved respectively to self.species[sp_name]['conc'] (list of np.arrays)")
            if dense_output:
                self.soln_continuous = [soln.sol for soln in solns] 
                self.logger.info(f"\tSaving list of continuous solution functions to self.soln_continuous (list of scipy.integrate.OdeSolution's)")
            else:
                self.soln_continuous = None
                self.logger.info("\tNot saving continuous solutions. Use dense_output=True to save them to self.soln_continuous")
        
        if output_raw:
            if conc0_mat_len == 1:
                self.logger.info("Returning raw solver output.\n")
                return solns[0]
            self.logger.info("Returning list of raw solver outputs.\n")        
            return solns
        else:
            self.logger.info("Not returning raw solver output. Use output_raw=True to return raw data.\n")
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
        #TODO: show pop dist is like odes
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

    def conc_mat2dict(self,conc_mat):
        """
        Save species vectors from a concentration matrix to the respective species[name]['conc'] based on species[name]['index'].
        Useful for saving the output of a continuous solution to the species dictionary.
            Don't forget `system.t_dcdts = t`
        """
        for _, sp_data in self.species.items():
            sp_data['conc'] = conc_mat[sp_data['index']]
        return



