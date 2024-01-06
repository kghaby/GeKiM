import numpy as np
import re
import copy
import logging
import sys
from scipy.integrate import solve_ivp

class NState:
    #TODO: Add stochastic method
    
    def __init__(self, config, logfilename=None):
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
        self.logger.setLevel(logging.INFO)

        if logfilename:
            file_handler = logging.FileHandler(logfilename)
            #file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler(sys.stdout)
        #stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stream_handler)

        if not self.validate_config(config):
            raise ValueError("Invalid config data provided.")
        
        self.config = copy.deepcopy(config)
        self.species = self.config['species']
        self.species_order = {name: idx for idx, name in enumerate(self.config['species'])}
        self.transitions = self.config['transitions']
        self._preprocess_transitions()
        self.sol = None
        
        self.logger.info(f"NState model initialized successfully.")

    def validate_config(self, config):
        """
        Validate the provided configuration data.

        Parameters:
        config (dict): Configuration to validate.

        Returns:
        bool: True if valid, False otherwise.
        """
        #TODO: Make sure names are unique and nonempty 
        return 'species' in config and 'transitions' in config

    def _preprocess_transitions(self):
        """
        Preprocess the transitions by extracting coefficients and species names.
        """
        for _, tr in self.transitions.items():
            tr['from'] = [self.identify_coeff(s) for s in tr['from']]
            tr['to'] = [self.identify_coeff(s) for s in tr['to']]

    @staticmethod
    def identify_coeff(species_str):
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

    def dcdt(self, t, concentrations):
        """
        Compute the derivative of concentrations with respect to time.

        Parameters:
        t (float): Time.
        concentrations (numpy.ndarray): Array of species concentrations.

        Returns:
        numpy.ndarray: Array of concentration time derivatives.
        """
        dcdt_arr = np.zeros(len(self.species))
        for tr in self.transitions.values():
            rate_constant = tr['value']
            rate = rate_constant * np.prod([concentrations[self.species_order[sp_name]] ** coeff for coeff, sp_name in tr['from']])
            for coeff, sp_name in tr['from']:
                dcdt_arr[self.species_order[sp_name]] -= coeff * rate
            for coeff, sp_name in tr['to']:
                dcdt_arr[self.species_order[sp_name]] += coeff * rate
        return dcdt_arr

    def _log_dcdts(self):
        """
        Log the ordinary differential equations for the concentrations of each species over time in a readable format.
        """
        dcdt_dict = {}
        max_header_length = 0

        # Find the max length for the headers
        for sp_name in self.species:
            header_length = len(f"d{sp_name}/dt")
            max_header_length = max(max_header_length, header_length)

        # Write eqn headers and rate laws
        for sp_name in self.species:
            dcdt_dict[sp_name] = [f"d{sp_name}/dt".ljust(max_header_length) + " ="]

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

    def solve_ode(self, t):
        """
        Solve the ordinary differential equations for the system.

        Parameters:
        t (list or numpy.ndarray): Time points to solve the ODEs at.

        Raises:
        RuntimeError: If the ODE solver fails.
        """
        conc0 = np.array([sp['conc'] for _, sp in self.species.items()])
        t_span = (t[0], t[-1])
        self._log_dcdts()
        try:
            solution = solve_ivp(
                lambda t, conc: self.dcdt(t, conc),
                t_span, conc0, t_eval=t, method='BDF', rtol=1e-6, atol=1e-8
            )
            if not solution.success:
                raise RuntimeError("ODE solver failed: " + solution.message)
            self.sol = solution.y.T
            self.logger.info("ODEs solved successfully.")
        except Exception as e:
            self.logger.error(f"Error in solving ODEs: {e}")
            raise

    def gillespie_sim(self):
        pass  # Gillespie algorithm to be implemented