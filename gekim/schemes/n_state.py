import numpy as np
import re
import copy
import sys
from scipy.integrate import solve_ivp
from itertools import product
from sympy import symbols, Matrix, prod, pretty, zeros, lambdify
from ..utils import integerable_float,Logger
from ..simulators.gillespie import Gillespie
from ..simulators.simulator import Simulator
from typing import Callable, Any, Union


#TODO: find_linear_paths in kinetics2 and more general pathfinder in utils?
    
class Species:
    def __init__(self, name: str, conc: Union[np.ndarray,float], label=None, color=None, index=None, ode=None):
        """
        Recommended to initialize with the following arguments:
        name (str): Name of the species.
        conc (np.ndarray|float): Initial concentration of the species.
        label (str): Useful for plotting. Will default to NAME.
        color (str): useful for plotting. Best added by ..utils.Plotting.assign_colors_to_species().

        Args that should be left alone unless you know what you're doing:
        index (int): Index of the species in the system. Added by the NState class.
        ode (float): Symbolic rate law for the species. Added by the NState class.

        """
        self.name = name
        self.conc = np.array([conc]) if np.isscalar(conc) else np.array(conc)
        self.label = label or name
        self.index = index
        self.color = color
        self.ode = ode
        self.sym = symbols(name)
        self.prob = None # added by simulate

    def __repr__(self):
        return f"{self.name} (Concentration: {self.conc}, Label: {self.label})"

class Transition:
    def __init__(self, name: str, k: float, source: list, target: list, label=None, index=None):
        """
        Recommended to initialize with the following arguments:
        name (str): Name of the rate constant.
        k (float): Value of the rate constant.
        source (list): List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        target (list): List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        label (str): Could be useful for plotting. Will default to NAME.
    
        Args that should be left alone unless you know what you're doing:
        index (int): Index of the transition in the system. Added by the NState class.
        """
        self.name = name # should be the name of the rate constant for all intents and purposes, eg "kon"
        self.k = k
        self.source = Transition._format_transition(source,"source")  # List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        self.target = Transition._format_transition(target,"target")  # List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings
        self.label = label or name
        self.index = index
        self.sym = symbols(name)
        

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
    #TODO: config conc to conc0, .conc to .soln, .prob to .conc
    
    def __init__(self, config: dict, logfilename=None, quiet=False):
        """
        Initialize the NState class with configuration data.

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
                conc=np.array([data["conc"]]) if np.isscalar(data["conc"]) else np.array(data["conc"]),
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

        self.setup_calculations()
        self.log.info(f"NState system initialized successfully.\n")

    def setup_calculations(self):
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
        
        self._generate_matrices_for_odes_func() # generates self._unit_sp_mat, self._stoich_mat, self._stoich_reactant_mat, self._k_vec, self._k_diag

        sp_syms = {name: symbols(name) for name in self.species}
        tr_syms = {name: symbols(name) for name in self.transitions}

        # Rate laws (ode)
        self._generate_odes_sym(sp_syms,tr_syms) # generates self.odes_sym and self.odes_numk
        self.log_odes()
        tr_sym2num = {symbols(name): tr.k for name, tr in self.transitions.items()}
        self.odes_numk = self.odes_sym.subs(tr_sym2num)
        #self._lambdify_sym_odes(sp_syms) # Overwrites self._ode with a lambdified version of self.odes_sym
        self.t_odes = None 

        # Jacobian
        self._generate_jac(sp_syms) # generates self.J_sym, self.J_symsp_numtr, self.J_func_wrap
        self.log_jac()

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

    def _generate_odes_sym(self,sp_syms,tr_syms):
        """
        Generate symbolic rate laws for each species.
        """
        # Generate ODE's with symbolic species and rate constants 
        odes_sym = Matrix([0] * len(sp_syms))
        for tr_name, tr in self.transitions.items():
            unscaled_rate = tr_syms[tr_name] * prod(sp_syms[sp_name]**coeff for sp_name, coeff in tr.source)
            for sp_name, coeff in tr.source:
                odes_sym[self.species[sp_name].index] -= coeff * unscaled_rate
            for sp_name, coeff in tr.target:
                odes_sym[self.species[sp_name].index] += coeff * unscaled_rate
        self.odes_sym = odes_sym 

        # Assign each ode to the respective species
        for sp_name, sp_data in self.species.items():
            sp_data.ode = odes_sym[sp_data.index]
        self.log.info("Assigned symbolic ODE's to species (self.species[NAME].ode).\n")

        # Substitute rate constant symbols for values 
        tr_sym2num = {symbols(name): tr.k for name, tr in self.transitions.items()}
        self.odes_numk = self.odes_sym.subs(tr_sym2num)
        return 
    
    def _lambdify_sym_odes(self,sp_syms):
        """
        Convert the symbolic ode vector (with numerical rate constants) into a numerical function.
        This overwrites the native self._ode function. It's just as fast typically, if not a little faster.
        Not currently utilized, but this might be useful someday.
        """
        species_vec = Matrix([sp_syms[name] for name in self.species])
        odes_func = lambdify(species_vec, self.odes_numk, 'numpy')
        self._odes_func = lambda t, y: odes_func(*y).flatten()
        return

    def log_odes(self,force_print=False):
        """
        Log the symbolic ODE's.
        """
        ode_dict = {}
        max_header_length = 0

        # Find the max length for the headers
        for sp_name in self.species:
            header_length = len(f"d[{sp_name}]/dt")
            max_header_length = max(max_header_length, header_length)

        # Write eqn headers and rate laws
        for sp_name in self.species:
            ode_dict[sp_name] = [f"d[{sp_name}]/dt".ljust(max_header_length) + " ="]

        for tr_name, tr in self.transitions.items():
            # Write rate law
            rate = f"{tr_name} * " + " * ".join([f"{sp_name}^{coeff}" if coeff != 1 else f"{sp_name}" for sp_name,coeff in tr.source])
            rate = rate.rstrip(" *")  # Remove trailing " *"

            # Add rate law to the eqns
            for sp_name,coeff in tr.source:
                term = f"{coeff} * {rate}" if coeff != 1 else rate
                ode_dict[sp_name].append(f" - {term}")

            for sp_name,coeff in tr.target:
                term = f"{coeff} * {rate}" if coeff != 1 else rate
                ode_dict[sp_name].append(f" + {term}")

        # Construct the final string
        ode_log = "ODE's:\n\n"
        for sp_name, eqn_parts in ode_dict.items():
            # Aligning '+' and '-' symbols
            eqn_header = eqn_parts[0]
            terms = eqn_parts[1:]
            aligned_terms = [eqn_header + " " + terms[0]] if terms else [eqn_header]
            aligned_terms += [f"{'':>{max_header_length + 3}}{term}" for term in terms[1:]]
            formatted_eqn = "\n".join(aligned_terms)
            ode_log += formatted_eqn + '\n\n'

        self.log.info(ode_log)
        if force_print:
            print(ode_log)
        return

    def _generate_matrices_for_odes_func(self):
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
            tr_idx = tr.index
            self._k_vec[tr_idx] = tr.k
            reactant_vec = np.sum([self._unit_sp_mat[self.species[name].index] * coeff for name, coeff in tr.source],axis=0)
            product_vec = np.sum([self._unit_sp_mat[self.species[name].index] * coeff for name, coeff in tr.target],axis=0)
            
            self._stoich_reactant_mat[tr_idx, :] = reactant_vec  
            #self._stoich_product_mat[tr_idx, :] = product_vec   # not used
            self._stoich_mat[tr_idx] = product_vec - reactant_vec

        self._k_diag = np.diag(self._k_vec)

        return

    def _generate_jac(self,sp_syms):
        """
        Generate the symbolic Jacobian matrix and convert it to a numerical function.
        Saves the symbolic Jacobian to self.J_sym and the (wrapped) numerical function to self.J_func_wrap(t,y).
        Also saves a Jacobian with symbolic species and numeric transition rate constants to self.J_symsp_numtr.

        The wrapped numerical function is time-independent even though it accepts t as an argument!

        The Jacobian matrix here represents the first-order partial derivatives of the rate of change equations
        for each species with respect to all other species in the system.
        """
        
        species_vec = Matrix(list(sp_syms.values()))

        # Symbolic Jacobian
        self.J_sym = self.odes_sym.jacobian(species_vec)

        # Numerical Jacobian
        self.J_symsp_numtr = self.odes_numk.jacobian(species_vec) # Symbolic species, numeric transition rate constants
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
            J_log[0, sp_data.index+1] = symbols(sp_name)
            J_log[sp_data.index+1, 0] = symbols(f'd[{sp_name}]/dt')
        J_log[0,0] = symbols("_")

        J_log_str = "Jacobian (including row and column labels):\n"
        J_log_str += pretty((J_log),use_unicode=True)
        J_log_str += "\n"
        self.log.info(J_log_str)
        
        return

    def _odes_func(self, t, conc):
        """
        Cannot model rates that are not simple power laws (eg dynamic inhibition, cooperativity, time dependent params). 
        But most of these can be baked in on the schematic level I think. 
        """
        #TODO: Use higher dimensionality conc arrays to process multiple input concs at once? 
        C_Nr = np.prod(np.power(conc, self._stoich_reactant_mat), axis=1) # state dependencies
        N_K = np.dot(self._k_diag,self._stoich_mat) # interactions
        odes = np.dot(C_Nr,N_K)
        return odes

    def solve_odes(self, t_eval: np.ndarray = None, t_span: tuple = None, conc0_dict: dict = None, method='BDF', rtol=1e-6, atol=1e-8, 
                    output_raw=False, dense_output=False):
        """
        Solve the ODEs of species concentration wrt time for the system. 
        Will update self.species.conc with the respective solutions.

        Arguments:
        t_eval (np.array): Time points for ODE solutions.

        t_span (tuple): Time span for ODE solutions.

        conc0_dict (dict: {str: np.ndarray}): Dictionary of {species_name: conc0_arr} pairs for initial concentrations to simulate. 
            Unprovided species will use self.species[NAME].conc[0] as a single-point initial concentration.
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
            Access a specific species conc like soln_continuous(t)[self.species[NAME].index].

        """
        #TODO: check how combinations are arranged and make sure its intuitive to separate and use them (ie the indexing is clear)
        #TODO: More analytically approximate the time scale.
            # incorporate the network of transitions, nonlinearity, etc?
            # Linear scaling of the inverse min eigenvalue underestimates when conc0E ~= conc0I
            # Linearize system then use normal mode frequency of linear system (1/(sqrt(smallest eigenvalue))?
            # needs to be an n-dimensional function, where n is the degree of (non)linearity
        #TODO: smarter way to choose rtol and atol. M-scale kon and nM-scale conc cause wild issues that are resolved with changing rtol and atol from default
            # at least a warning if a weird solution is found 

        conc0_mat = self._make_conc0_mat(conc0_dict)
        conc0_mat_len = len(conc0_mat)
        self.log.info(f"Solving the timecourse from {conc0_mat_len} initial concentration vectors...")

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
                    self.log.info(f"\tEstimated time scale: {naive_time_scale:.2e} (1/<rate constant units>)")
                    
                else:
                    t_span = (t_eval[0], t_eval[-1])

            soln = solve_ivp(self._odes_func, t_span=t_span, y0=conc0, method=method, t_eval=t_eval, 
                                rtol=rtol, atol=atol, jac=self.J_func_wrap, dense_output=dense_output) 
                # vectorized=True makes legacy ode func slower bc low len(conc0) I think
            if not soln.success:
                raise RuntimeError("FAILED: " + soln.message)
            solns.append(soln)
            
        self.log.info("ODEs solved successfully. Saving data...")

        if conc0_mat_len == 1:
            self.t_odes = soln.t
            self.log.info(f"\tTime saved to self.t_odes (np.array)")
            for _, data in self.species.items():
                data.conc = soln.y[data.index]
            self.log.info(f"\tConcentrations saved respectively to self.species[sp_name].conc (np.array)")
            if dense_output:
                self.soln_continuous = soln.sol
                self.log.info(f"\tSaving continuous solution function to self.soln_continuous(t) (scipy.integrate.OdeSolution)")
            else:
                self.soln_continuous = None
                self.log.info("\tNot saving continuous solution. Use dense_output=True to save it to self.soln_continuous")
        else:
            self.t_odes = [soln.t for soln in solns] 
            self.log.info(f"\t{conc0_mat_len} time vectors saved to self.t_odes (list of np.arrays)")
            for _, data in self.species.items():
                data.conc = [soln.y[data.index] for soln in solns]
            self.log.info(f"\t{conc0_mat_len} concentration vectors saved respectively to self.species[sp_name].conc (list of np.arrays)")
            if dense_output:
                self.soln_continuous = [soln.sol for soln in solns] 
                self.log.info(f"\tSaving list of continuous solution functions to self.soln_continuous (list of scipy.integrate.OdeSolution's)")
            else:
                self.soln_continuous = None
                self.log.info("\tNot saving continuous solutions. Use dense_output=True to save them to self.soln_continuous")
        
        if output_raw:
            if conc0_mat_len == 1:
                self.log.info("Returning raw solver output.\n")
                return solns[0]
            self.log.info("Returning list of raw solver outputs.\n")        
            return solns
        else:
            self.log.info("Not returning raw solver output. Use output_raw=True to return raw data.\n")
            return
    
    def _make_conc0_mat(self,conc0_dict=None):
        if conc0_dict:
            combinations = product(*(
                np.atleast_1d(conc0_dict.get(sp_name, [np.atleast_1d(sp_data.conc).flatten()[0]])) 
                for sp_name, sp_data in self.species.items()
            ))
            conc0_mat = np.vstack([comb for comb in combinations])
        else:
            conc0_mat = np.atleast_2d([np.atleast_1d(sp_data.conc).flatten()[0] for _, sp_data in self.species.items()])
        return conc0_mat

    def sum_conc(self,whitelist:list=None,blacklist:list=None):
        """
        Sum the concentrations of specified species in the model.
        Whitelist and blacklist cannot be provided simultaneously.

        Args:
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

        total_concentration = np.sum([self.species[name].conc for name in species_names], axis=0)

        return total_concentration

    def conc_mat2dict(self,conc_mat):
        """
        Save species vectors from a concentration matrix to the respective species[NAME].conc based on species[NAME].index.
        Useful for saving the output of a continuous solution to the species dictionary.
            Don't forget `system.t_odes = t`
        """
        for _, sp_data in self.species.items():
            sp_data.conc = conc_mat[sp_data.index]
        return



        #TODO: Although i do like the idea of being able to continue from a previous run, so add an option called "continue" which takes an integer which points to the index of the run that its continuing from
        #TODO: show pop dist is like odes
        # Initialize
        #TODO: S2.alt1 breaks this from negative rates somehow
    #TODO: running prob? and test more
    #TODO: type hints on algo
    #TODO: 2S I goes neg

    def _process_results(self, results, output_raw):
        if len(results) == 1:
            result = results[0]
            self.t_sim = result['t']
            self.prob_dist = result['prob_dist']
            for sp_name, sp_data in self.species.items():
                sp_data.prob = self.prob_dist[:, sp_data.index]
            return result if output_raw else None
        else:
            # Handle multiple initial conditions
            self.t_sim = [result['t'] for result in results]
            self.prob_dist = [result['prob_dist'] for result in results]
            for idx, result in enumerate(results):
                for sp_name, sp_data in self.species.items():
                    sp_data.prob = result['prob_dist'][:, sp_data.index]
            return results if output_raw else None

    
    def simulate(self, t_max, conc0_dict=None, num_replicates=1, output_times=None, output_raw=False, simulator: Simulator = Gillespie,**kwargs):
        if not isinstance(simulator, Simulator):
            simulator = simulator(self)  # Instantiate if the argument passed is a class, not an instance
        conc0_mat = self._make_conc0_mat(conc0_dict)
        results = [simulator.simulate(t_max, conc0, num_replicates, output_times,**kwargs) for conc0 in conc0_mat]
        return self._process_results(results, output_raw)
                


