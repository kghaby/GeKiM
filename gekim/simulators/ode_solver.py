import numpy as np
from scipy.integrate import solve_ivp
from sympy import symbols, Matrix, prod, pretty, zeros, lambdify
from .base_simulator import BaseSimulator

class ODESolver(BaseSimulator):
    """
    Parameters
    ----------
    system : NState
        The system object.

    Notes
    -----
    The input NState instance, `system`, will be modified directly by the simulator, 
    whether the simulator is added as an attribute to `system` or not. 

    Use as a simulator for NState class. Will generate and solve systems deterministically.

    During initialization, the following keys (and values) are added to the 
    `self.system.simin` dict:

        unit_sp_mat : np.ndarray
            Unit species matrix.
        stoich_mat : np.ndarray
            Stoichiometry matrix.
        stoich_reactant_mat : np.ndarray
            Stoichiometry reactant matrix.
        k_vec : np.ndarray
            Rate constant vector.
        k_diag : np.ndarray
            Diagonal matrix of rate constants.
        rates_sym : sympy.Matrix
            Species rate vectors with symbolic species and rate constants.
        rates_numk : sympy.Matrix
            Species rate vector with symbolic species and numerical rate constants.
        J_sym : sympy.Matrix
            Jacobian matrix with symbolic species and rate constants.
        J_symsp_numtr : sympy.Matrix
            Jacobian matrix with symbolic species and numerical rate constants.
        J_func_wrap : Callable
            Wrapped numerical Jacobian function that accepts t, y as arguments.
    """

    def setup(self):
        """
        (Re)initialize the simulator by generating the necessary matrices and symbolic expressions, 
        and saving them to the system and species simin dictionaries.
        """
        
        self._generate_matrices_for_rates_func() 

        # Rate laws 
        self._generate_rates_sym() 
        self.log_rates()
        #self.lambdify_sym_rates() # Overwrites self._rates_func with a lambdified version of self.system.simin["rates_sym"]

        # Jacobian
        self._generate_jac() 
        self.log_jac()

    def _generate_matrices_for_rates_func(self):
        """
        Notes
        -----
        Generates 
        - unit species matrix (`self.system.simin["unit_sp_mat"]`), 
        - stoichiometry matrix (`self.system.simin["stoich_mat"]`), 
        - stoichiometry reactant matrix (`self.system.simin["stoich_reactant_mat"]`), and 
        - rate constant vector (`self.system.simin["k_vec"]`) 
            and diagonal matrix (`self.system.simin["k_diag"]`).

        Rows are transitions, columns are species.

        Used for solving rates.
        """

        n_species = len(self.system.species)
        n_transitions = len(self.system.transitions)
        self.system.simin["unit_sp_mat"] = np.eye(n_species, dtype=int)

        # Initialize matrices
        self.system.simin["stoich_reactant_mat"] = np.zeros((n_transitions, n_species))
        self.system.simin["stoich_mat"] = np.zeros((n_transitions, n_species))
        self.system.simin["k_vec"] = np.zeros(n_transitions)

        # Fill stoich matrices and k vector
        for tr_name, tr in self.system.transitions.items():
            tr_idx = tr.index
            self.system.simin["k_vec"][tr_idx] = tr.k
            reactant_vec = np.sum([self.system.simin["unit_sp_mat"][self.system.species[name].index] * coeff for name, coeff in tr.source],axis=0)
            product_vec = np.sum([self.system.simin["unit_sp_mat"][self.system.species[name].index] * coeff for name, coeff in tr.target],axis=0)
            
            self.system.simin["stoich_reactant_mat"][tr_idx, :] = reactant_vec  
            #self._stoich_product_mat[tr_idx, :] = product_vec   # not used
            self.system.simin["stoich_mat"][tr_idx] = product_vec - reactant_vec

        self.system.simin["k_diag"] = np.diag(self.system.simin["k_vec"])

        return
    
    def _rates_func(self, t, conc):
        """
        Calculate the rate vector at time t and concentration vector conc.

        Notes
        -----
        Cannot model rates that are not simple power laws (eg dynamic inhibition, cooperativity, time dependent params). 
        But most of these can be baked in on the schematic level I think. 
        """
        #TODO: Use higher dimensionality conc arrays to process multiple input concs at once? 
        C_Nr = np.prod(np.power(conc, self.system.simin["stoich_reactant_mat"]), axis=1) # state dependencies
        N_K = np.dot(self.system.simin["k_diag"],self.system.simin["stoich_mat"]) # interactions
        rates = np.dot(C_Nr,N_K)
        return rates
    
    def _generate_rates_sym(self):
        """
        Notes
        -----
        Generates 
        - Species rate vectors with symbolic species and rate constants 
        - Species rate vectors with symbolic species and numerical rate constants

        Adds these key/value pairs to both system and species simin dictionaries.
        """
        # Generate rate's with symbolic species and rate constants 
        rates_sym = Matrix([0] * len(self.system.species))
        for tr_name, tr in self.system.transitions.items():
            unscaled_rate = self.system.transitions[tr_name].sym * prod(self.system.species[sp_name].sym**coeff for sp_name, coeff in tr.source)
            for sp_name, coeff in tr.source:
                rates_sym[self.system.species[sp_name].index] -= coeff * unscaled_rate
            for sp_name, coeff in tr.target:
                rates_sym[self.system.species[sp_name].index] += coeff * unscaled_rate
        self.system.simin["rates_sym"] = rates_sym 
        self.system.log.info("Saved rates_sym and rates_numk into system simin dict (SYSTEM.simin).")

        # Substitute rate constant symbols for values 
        tr_sym2num = {symbols(name): tr.k for name, tr in self.system.transitions.items()}
        self.system.simin["rates_numk"] = self.system.simin["rates_sym"].subs(tr_sym2num)

        # Assign each rate law to the respective species
        for sp_name, sp_data in self.system.species.items():
            sp_data.simin["rate_sym"] = self.system.simin["rates_sym"][sp_data.index]
            sp_data.simin["rate_numk"] = self.system.simin["rates_numk"][sp_data.index]
        self.system.log.info("Saved rate_sym and rate_numk into species simin dict (SYSTEM.species[NAME].simin).")

        return 
    
    def lambdify_sym_rates(self):
        """
        Convert the symbolic rate vector (with numerical rate constants) into a numerical function.
    
        Notes
        -----
        This overwrites the unsymbolic self._rates_func function. It is slightly slower but more flexible.

        Can be used to simulate with custom symbolic rate funcs for each species instead of all following the same ODE paradigm.
        """
        #TODO: test custom funcs and refactor as needed to make sure its easier and more robust to use (ie make sure the appropriate setup steps are reran
        # currently uses the system.simin, not species specific. would need to remake system.simin, or alter rate func at the generate_rates_sym step
        species_vec = Matrix([self.system.species[sp_name].sym for sp_name in self.system.species])
        rates_func = lambdify(species_vec, self.system.simin["rates_numk"], 'numpy')
        self._rates_func = lambda t, y: rates_func(*y).flatten()

        # Re-generate the Jacobian matrix in case this function is called after the Jacobian has been generated, like for custom functions
        #self.generate_jac()
        return

    def log_rates(self,force_print=False):
        """
        Log the symbolic rates using the system's logger.

        Parameters
        ----------
        force_print : bool, optional
            If True, the rates will be printed to the console. Default is False.
        """
        rate_dict = {}
        max_header_length = 0

        # Find the max length for the headers
        for sp_name in self.system.species:
            header_length = len(f"d[{sp_name}]/dt")
            max_header_length = max(max_header_length, header_length)

        # Write eqn headers and rate laws
        for sp_name in self.system.species:
            rate_dict[sp_name] = [f"d[{sp_name}]/dt".ljust(max_header_length) + " ="]

        for tr_name, tr in self.system.transitions.items():
            # Write rate law
            rate = f"{tr_name} * " + " * ".join([f"{sp_name}^{coeff}" if coeff != 1 else f"{sp_name}" for sp_name,coeff in tr.source])
            rate = rate.rstrip(" *")  # Remove trailing " *"

            # Add rate law to the eqns
            for sp_name,coeff in tr.source:
                term = f"{coeff} * {rate}" if coeff != 1 else rate
                rate_dict[sp_name].append(f" - {term}")

            for sp_name,coeff in tr.target:
                term = f"{coeff} * {rate}" if coeff != 1 else rate
                rate_dict[sp_name].append(f" + {term}")

        # Construct the final string
        rate_log = "\nRates:\n\n"
        for sp_name, eqn_parts in rate_dict.items():
            # Aligning '+' and '-' symbols
            eqn_header = eqn_parts[0]
            terms = eqn_parts[1:]
            aligned_terms = [eqn_header + " " + terms[0]] if terms else [eqn_header]
            aligned_terms += [f"{'':>{max_header_length + 3}}{term}" for term in terms[1:]]
            formatted_eqn = "\n".join(aligned_terms)
            rate_log += formatted_eqn + '\n\n'

        self.system.log.info(rate_log)
        if force_print:
            print(rate_log)
        return

    def _generate_jac(self):
        """
        Notes
        -----
        Generates
         - the Jacobian matrix with symbolic species and rate constants (self.system.simin["J_sym"])
         - the Jacobian matrix with symbolic species and numerical rate constants (self.system.simin["J_symsp_numtr"])
         - a wrapped numerical Jacobian function (self.system.simin["J_func_wrap"]) that accepts t,y as arguments

        The wrapped numerical function is time-independent even though it accepts t as an argument!

        The Jacobian matrix here represents the first-order partial derivatives of the rate of 
            change equations for each species with respect to all other species in the system.
        """
        
        species_vec = Matrix([self.system.species[sp_name].sym for sp_name in self.system.species])

        # Symbolic Jacobian
        self.system.simin["J_sym"] = self.system.simin["rates_sym"].jacobian(species_vec)

        # Numerical Jacobian
        self.system.simin["J_symsp_numtr"] = self.system.simin["rates_numk"].jacobian(species_vec) # Symbolic species, numeric transition rate constants
        J_func = lambdify(species_vec, self.system.simin["J_symsp_numtr"], 'numpy') # Make numerical function
        self.system.simin["J_func_wrap"] = lambda t, y: J_func(*y) # Wrap J_func so that t,y is passed to the function to be compatible with solve_ivp
        return
      
    def log_jac(self,force_print=False):
        """
        Log the symbolic representation of the Jacobian matrix using the system's logger.

        Notes
        -----
        The logged Jacobian includes row and column labels, but `SYSTEM.simin["J_sym"]` does not.

        The Jacobian matrix here represents the first-order partial derivatives of the rate of 
            change equations for each species with respect to all other species in the system.
        """
        n_species = len(self.system.species)
        J_log = zeros(n_species + 1, n_species + 1)
        J_log[1:, 1:] = self.system.simin["J_sym"]

        for sp_name,sp_data in self.system.species.items():
            J_log[0, sp_data.index+1] = symbols(sp_name)
            J_log[sp_data.index+1, 0] = symbols(f'd[{sp_name}]/dt')
        J_log[0,0] = symbols("_")

        J_log_str = "Jacobian (including row and column labels):\n"
        J_log_str += pretty((J_log),use_unicode=False)
        J_log_str += "\n"
        self.system.log.info(J_log_str)
        if force_print:
            print(J_log_str)
        return

    def simulate(self, t_eval: np.ndarray = None, t_span: tuple = None, method='BDF', rtol=1e-3, atol=0, dense_output=False, output_raw=False, **kwargs):
        """
        Solve the differential equations of species concentration wrt time for the system. 

        Parameters
        ----------
        output_raw : bool, optional
            If True, return raw solver output.
        t_eval : np.ndarray, optional
            Time points for rate solutions.
        t_span : tuple, optional
            Time span for rate solutions.
        method : str, optional
            Integration method, default is 'BDF'.
        rtol : float, optional
            Relative tolerance for the solver. Default is 1e-3.
        atol : float, optional
            Absolute tolerance for the solver. Default is 0. 
            If atol == 0, atol = 1e-6 * <the smallest nonzero value of y0>.
                Dynamically chosen per y0.
        dense_output : bool, optional
            If True, save a `scipy.integrate._ivp.common.OdeSolution` instance to `SYSTEM.simout.soln_continuous(t)`.
            If using multiple y0's, this will be a list of instances that share indexing with the other outputs,
            and can be called like `SYSTEM.simout['soln_continuous'][idx](t)`.
            Access a specific species conc like `SYSTEM.simout['soln_continuous'](t)[SYSTEM.species[NAME].index]`.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the solver.

        Notes
        -----
        Saves the following keys to `SYSTEM.simout`:
        - t : np.ndarray
            Time points
        - soln_continuous : scipy.integrate._ivp.common.OdeSolution or list of scipy.integrate._ivp.common.OdeSolution
            Continuous solution function (if `dense_output=True`)

        Saves the following keys to each species' `simout` dictionary attribute:
        - y : np.ndarray
            Concentration vectors

        Returns
        -------
        None or scipy.integrate._ivp.ivp.OdeResult or list of scipy.integrate._ivp.ivp.OdeResult
        """
        #TODO: check how combinations are arranged and make sure its intuitive to separate and use them (ie the indexing is clear)
        #TODO: More analytically approximate the time scale.
            # incorporate the network of transitions, nonlinearity, etc?
            # Linear scaling of the inverse min eigenvalue underestimates when y0E ~= y0I
            # Linearize system then use normal mode frequency of linear system (1/(sqrt(smallest eigenvalue))?
            # needs to be an n-dimensional function, where n is the degree of (non)linearity

        y0_mat = self._make_y0_mat()
        y0_mat_len = len(y0_mat)
        self.system.log.info(f"Solving the timecourse from {y0_mat_len} initial concentration vectors...")

        input_atol = atol
        input_t_eval = t_eval
        input_t_span = t_span
        solns = [None]*y0_mat_len
        for i,y0 in enumerate(y0_mat):
            if input_atol == 0:
                min_y0 = np.min(y0[y0 != 0])
                atol = min_y0*1e-6 # smallest nonzero value of the Jacobian
                self.system.log.info(f"\tSetting atol to {atol:.2e}, 1e-6 * <the smallest nonzero value of y0 ({min_y0})> .")
                
            if input_t_span is None:
                if input_t_eval is None:
                    J0 = self.system.simin["J_func_wrap"](None, y0)
                    t_span = self.estimate_t_span(J0)
                    self.system.log.info(f"\tEstimated time scale: {t_span[1]:.2e} (1/<rate constant units>)")
                else:
                    t_span = (t_eval[0], t_eval[-1])

            soln = solve_ivp(self._rates_func, t_span=t_span, y0=y0, method=method, t_eval=t_eval, 
                                rtol=rtol, atol=atol, jac=self.system.simin["J_func_wrap"], dense_output=dense_output, **kwargs) 
                # vectorized=True makes legacy rate func slower bc low len(y0) I think
            if not soln.success:
                raise RuntimeError("FAILED: " + soln.message)
            solns[i] = soln
            
        self.system.log.info("ODE's solved successfully. Saving data...")

        self._process_simouts(solns,y0_mat_len,dense_output=dense_output)

        if output_raw:
            if y0_mat_len == 1:
                self.system.log.info("Returning raw solver output.\n")
                return solns[0]
            self.system.log.info("Returning list of raw solver outputs.\n")        
            return solns
        else:
            self.system.log.info("Not returning raw solver output. Use output_raw=True to return raw data.\n")
            return

    @staticmethod
    def estimate_t_span(J0: np.ndarray) -> tuple[float, float]:
        """
        Estimate the timespan needed for convergence based on the 
        smallest magnitude of the Jacobian eigenvalues at initial conditions
        
        Parameters:
        ----------
        J0 : np.ndarray
            The Jacobian of the initial conditions of the system
        
        Returns:
        -------
        tuple[float, float]
            A tuple containing the estimated timespan needed for convergence.
            The first element of the tuple is the start time (0) and the second
            element is the estimated end time based on the smallest magnitude
            of the Jacobian eigenvalues at the initial conditions.
        
        Raises:
        ------
        ValueError
            If no eigenvalues above the threshold are found, indicating that
            the time scale cannot be estimated.
        """
        eigenvalues = np.linalg.eigvals(J0)
        eigenvalue_threshold = 1e-6 # below 1e-6 is considered insignificant. float32 cutoff maybe
        filtered_eigenvalues = eigenvalues[np.abs(eigenvalues) > eigenvalue_threshold] 
        if filtered_eigenvalues.size == 0:
            raise ValueError("No eigenvalues above the threshold, unable to estimate time scale.")
        naive_time_scale = 1 / (np.abs(filtered_eigenvalues).min())
        est_time_scale = naive_time_scale * 6.5
        est_t_span = (0, est_time_scale) # Start at 0 or np.abs(filtered_eigenvalues).min()?
        return est_t_span
    
    def _process_simouts(self, raw_simouts: list, y0_mat_len: int, dense_output=False):
        if y0_mat_len == 1:
            self.system.simout["t"] = raw_simouts[0].t
            self.system.log.info(f"\tTime saved to SYSTEM.simout['t'] (np.array)")
            self.system.simout["y"] = raw_simouts[0].y
            for _, data in self.system.species.items():
                data.simout["y"] = self.system.simout["y"][data.index]
            self.system.log.info(f"\tConcentrations saved respectively to SYSTEM.species[NAME].simout['y'] (np.array)")
            if dense_output:
                self.system.simout["soln_continuous"] = raw_simouts[0].sol
                self.system.log.info(f"\tSaving continuous solution function to SYSTEM.simout['soln_continuous'](t) (scipy.integrate.ODESolution)")
            else:
                self.system.simout["soln_continuous"] = None
                self.system.log.info("\tNot saving continuous solution. Use dense_output=True to save it to SYSTEM.simout['soln_continuous']")
        else:
            self.system.simout["t"] = [raw_simout.t for raw_simout in raw_simouts] 
            self.system.log.info(f"\t{y0_mat_len} time vectors saved to SYSTEM.simout['t'] (list of np.arrays)")
            self.system.simout["y"] = [raw_simout.y for raw_simout in raw_simouts]
            for _, data in self.system.species.items():
                data.simout["y"] = [y[data.index] for y in self.system.simout["y"]]
            self.system.log.info(f"\t{y0_mat_len} concentration vectors saved respectively to SYSTEM.species[NAME].simout['y'] (list of np.arrays)")
            if dense_output:
                self.system.simout["soln_continuous"] = [raw_simout.sol for raw_simout in raw_simouts] 
                self.system.log.info(f"\tSaving list of continuous solution functions to SYSTEM.simout['soln_continuous'] (list of scipy.integrate.ODESolution's)")
            else:
                self.system.simout["soln_continuous"] = None
                self.system.log.info("\tNot saving continuous solutions. Use dense_output=True to save them to SYSTEM.simout['soln_continuous']")
        return
