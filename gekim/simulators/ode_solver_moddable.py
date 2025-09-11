import numpy as np
from scipy.integrate import solve_ivp
from sympy import Matrix, prod
import inspect

from .base import BaseSimulator

class ODESolverMod(BaseSimulator):
    """
    Supports callable rate constants.
    
    Parameters
    ----------
    system : NState
        The system object.
    """

    def setup(self):
        """
        (Re)initialize the simulator by generating the necessary matrices and symbolic expressions, 
        and saving them to the system and species simin dictionaries.
        """
    
        # Rate laws 
        self._generate_rates_sym() 
        self.log_rates()
    
    
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
            unscaled_rate = tr.sym * prod(self.system.species[sp_name].sym**coeff for sp_name, coeff in tr.source)
            for sp_name, coeff in tr.source:
                rates_sym[self.system.species[sp_name].index] -= coeff * unscaled_rate
            for sp_name, coeff in tr.target:
                rates_sym[self.system.species[sp_name].index] += coeff * unscaled_rate
        self.simin["rates_sym"] = rates_sym 
        self.system.log.info("Saved rates_sym and rates_numk into system simin dict (simin).")

        # Substitute rate constant symbols for values 
        subs = {}
        for tr_name, tr in self.system.transitions.items():
            if not callable(tr.k):
                subs[tr.sym] = tr.k
        self.simin["rates_numk"] = rates_sym.subs(subs)

        # Assign each rate law to the respective species
        for sp_name, sp_data in self.system.species.items():
            sp_data.simin["rate_sym"] = self.simin["rates_sym"][sp_data.index]
            sp_data.simin["rate_numk"] = self.simin["rates_numk"][sp_data.index]
        self.system.log.info("Saved rate_sym and rate_numk into species simin dict (SYSTEM.species[NAME].simin).")
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
        rate_log = "\nMass-action rates:\n\n"
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
    
    def _rates_func(self, t, y):
        # TODO: support params passed through simulate()
        dy = np.zeros(len(self.system.species))
        for tr in self.system.transitions.values():

            if callable(tr.k):
                # detect params
                sig = inspect.signature(tr.k)
                kwargs = {}
                if "t" in sig.parameters: kwargs["t"] = t
                if "y" in sig.parameters: kwargs["y"] = y
                k = tr.k(**kwargs)
            else:
                k = tr.k

            unscaled_rate = k * np.prod([y[self.system.species[sp_name].index]**coeff for sp_name, coeff in tr.source])
            for sp_name, coeff in tr.source:
                dy[self.system.species[sp_name].index] -= coeff * unscaled_rate
            for sp_name, coeff in tr.target:
                dy[self.system.species[sp_name].index] += coeff * unscaled_rate
        return dy

    def simulate(self, t_eval: np.ndarray = None, t_span: tuple = None, method='RK45', rtol=1e-3, atol=0, dense_output=False, output_raw=False, **kwargs):
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
            Integration method, default is 'LSODA'.
        rtol : float, optional
            Relative tolerance for the solver. Default is 1e-3.
        atol : float, optional
            Absolute tolerance for the solver. Default is 0. 
            If atol == 0, atol = 1e-6 * <the smallest nonzero value of y0>.
                Dynamically chosen per y0.
        dense_output : bool, optional
            If True, save a `scipy.integrate._ivp.common.OdeSolution` instance to `simout.soln_continuous(t)`.
            If using multiple y0's, this will be a list of instances that share indexing with the other outputs,
            and can be called like `simout['soln_continuous'][idx](t)`.
            Access a specific species conc like `simout['soln_continuous'](t)[SYSTEM.species[NAME].index]`.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the solver.

        Notes
        -----
        Saves the following keys to `simout`:
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
        y0_mat = self._make_y0_mat() # move to intialization unless the y0 has been changed? 
        y0_mat_len = len(y0_mat)
        self.system.log.info(f"Solving the timecourse from {y0_mat_len} initial concentration vectors...")

        input_atol = atol
        input_t_eval = t_eval
        input_t_span = t_span
        solns = [None]*y0_mat_len
        for i,y0 in enumerate(y0_mat):
            if np.any(y0 < 0):
                self.system.log.warning(f"WARNING: Negative y0 in y0_mat[{i}] = {y0}")

            if input_atol == 0:
                min_y0 = np.abs(np.min(y0[y0 != 0]))
                atol = min_y0*1e-6 
                self.system.log.info(f"\tSetting atol to {atol:.2e}, 1e-6 * <the absolute value of the smallest nonzero value of y0 ({min_y0})> .")
                
            if input_t_span is None:
                if input_t_eval is None:
                    t_span = (0, 1e5)  # Default time span if not provided
                    self.system.log.info(f"\tUsing default t_span: {t_span}")
                else:
                    t_span = (t_eval[0], t_eval[-1])

            soln = solve_ivp(self._rates_func, t_span=t_span, y0=y0, method=method, 
                             t_eval=t_eval, 
                             rtol=rtol, atol=atol,
                             dense_output=dense_output, **kwargs) 

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
    
    def _process_simouts(self, raw_simouts: list, y0_mat_len: int, dense_output=False):
        if y0_mat_len == 1:
            self.simout["t"] = raw_simouts[0].t
            self.system.log.info("\tTime saved to simout['t'] (np.array)")
            self.simout["y"] = raw_simouts[0].y
            for _, data in self.system.species.items():
                data.simout["y"] = self.simout["y"][data.index]
            self.system.log.info("\tConcentrations saved respectively to SYSTEM.species[NAME].simout['y'] (np.array)")
            if dense_output:
                self.simout["soln_continuous"] = raw_simouts[0].sol
                self.system.log.info("\tSaving continuous solution function to simout['soln_continuous'](t) (scipy.integrate.ODESolution)")
            else:
                self.simout["soln_continuous"] = None
                self.system.log.info("\tNot saving continuous solution. Use dense_output=True to save it to simout['soln_continuous']")
        else:
            self.simout["t"] = [raw_simout.t for raw_simout in raw_simouts] 
            self.system.log.info(f"\t{y0_mat_len} time vectors saved to simout['t'] (list of np.arrays)")
            self.simout["y"] = [raw_simout.y for raw_simout in raw_simouts]
            for _, data in self.system.species.items():
                data.simout["y"] = [y[data.index] for y in self.simout["y"]]
            self.system.log.info(f"\t{y0_mat_len} concentration vectors saved respectively to SYSTEM.species[NAME].simout['y'] (list of np.arrays)")
            if dense_output:
                self.simout["soln_continuous"] = [raw_simout.sol for raw_simout in raw_simouts] 
                self.system.log.info("\tSaving list of continuous solution functions to simout['soln_continuous'] (list of scipy.integrate.ODESolution's)")
            else:
                self.simout["soln_continuous"] = None
                self.system.log.info("\tNot saving continuous solutions. Use dense_output=True to save them to simout['soln_continuous']")
        return

