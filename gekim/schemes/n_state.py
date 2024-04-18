import numpy as np
import re
import copy
import logging
import sys
from scipy.integrate import solve_ivp
        
class NState:
    #TODO: Make sure its all np arrays and not lists 
    #TODO: Add stochastic method
    #TODO: logger retains previous classes? Jupyter output was showing previous class logs i think
    
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
        self._construct_matrices()
        self._construct_ode_mat()
        self.Y = np.abs(self._stoich_mat.clip(max=0))
        self.t = None
        
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
                
    @staticmethod
    def _parse_species_string(species_str):
        """
        Extract coefficient and species name from species string.

        Parameters:
        species_str (str): A species string, e.g., '2A'.

        Returns:
        tuple: A tuple of species name (str) and stoicheometric coefficient (int).
        """
        match = re.match(r"(\d*)(\D.*)", species_str)
        if match and match.groups()[0]:
            if float(int(float(match.groups()[0])) == float(match.groups()[0])): # Lets "5.0" be a valid stoich coeff
                coeff = int(match.groups()[0])
            else:
                raise ValueError(f"Invalid coefficient '{match.groups()[0]}' in species string '{species_str}'.")
        else:
            coeff = 1
        name = match.groups()[1] if match else species_str
        return name,coeff
                
    def _preprocess_transitions(self):
        """
        Preprocess the transitions by extracting coefficients and species names.
        """
        for _, tr in self.transitions.items():
            tr['from_idx'] = [(self.species[name]['index'], coeff) for name, coeff in (self._parse_species_string(sp) for sp in tr['from'])]
            tr['from'] = [(name, coeff) for name, coeff in (self._parse_species_string(sp) for sp in tr['from'])]
            tr['to_idx'] = [(self.species[name]['index'], coeff) for name, coeff in (self._parse_species_string(sp) for sp in tr['to'])]
            tr['to'] = [(name, coeff) for name, coeff in (self._parse_species_string(sp) for sp in tr['to'])]

    def _construct_matrices(self):
        n_species = len(self.species)
        n_transitions = len(self.transitions)
        self._stoich_mat = np.zeros((n_transitions, n_species), dtype=int)  # Stoichiometry coefficients
        self._k_vec = np.zeros(n_transitions)  # Rate constants

        sp_idx = {name: idx for idx, name in enumerate(self.species)}

        for tr_idx, (_,tr) in enumerate(self.transitions.items()):
            self._k_vec[tr_idx] = tr['value']
            for sp,coeff in tr['from']:
                self._stoich_mat[tr_idx,sp_idx[sp]] -= coeff
            for sp,coeff in tr['to']:
                self._stoich_mat[tr_idx,sp_idx[sp]] += coeff

        self._k_vec = np.diag(self._k_vec)
        self._stoich_mat=self._stoich_mat.T



    def dcdt2(self, t, concentrations):
        reaction_rates = self.rate_constants * np.prod(np.power(np.maximum(concentrations, 0), self.stoich_mat.T.clip(min=0)), axis=1)
        print(reaction_rates)
        return self.stoich_mat @ reaction_rates

    def _dcdt5(self, t, conc):
        #TODO: transpose others instead of currently transposed. Construct stoich from complex vectors 
        C = conc[:, np.newaxis]
        CY = np.prod(np.power(C, self.Y), axis=0)
        NK = np.dot(self._stoich_mat,self._k_vec)
        dCdt = np.dot(NK,CY)
        return dCdt
        #return np.dot(np.dot(self._stoich_mat,self._k_vec),np.prod(np.power(conc[:, np.newaxis], self.Y), axis=0))

    def _dcdt4(self, t, conc):
        
        R = self._stoich_mat.T
        K = self._k_vec
        X = conc[:, np.newaxis]
        Y = np.abs(R.clip(max=0))
        print(R)
        print(X)
        #print(Y)
        XY=np.prod(np.power(X, Y), axis=0)
        #print(XY)
        RK = np.dot(R,np.diag(K))
        #print(RK)
        dXdt=np.dot(RK,XY)
        #print(dXdt)
        raise ValueError

        #print(self._k_vec)
        #print(np.diag(self._k_vec))
        #print(self._stoich_mat)
        #print(self._k_vec*self._stoich_mat)
        mat=np.dot(np.diag(self._k_vec),self._stoich_mat) # produces ode_mat on linear system 
        mat = np.array([    [ 0.,          0.        ,  0.        ],\
                            [ 0.,         -0.02222222,  0.01111111],\
                            [ 0.,          0.00222222, -0.00111111]])
        print(mat) 
        #print(np.sum(mat,axis=0))
        #print(np.dot(self._k_vec,self._stoich_mat)) # produces collapsed ode_mat
        #print(self._stoich_mat)
        #print(conc)
        C = np.power(conc, np.abs(self._stoich_mat.clip(max=0)))
        print(C)
        #print(mat*C)
        print(np.dot(C,mat))

        #M=self._stoich_mat*np.power(conc, np.abs(self._stoich_mat))
        #print("M",M)
        #print(np.dot(self._k_vec,M))
        #print(np.dot(np.diag(self._k_vec),M))

        #print(np.power(conc, np.abs(self._stoich_mat)))
        #print(self._ode_mat)
        #print(np.sum(self._ode_mat,axis=0))
        #print(self._ode_mat.T * np.power(conc, np.abs(self._stoich_mat)))

        #print(np.sum(self._ode_mat.T * np.power(conc, np.abs(self._stoich_mat)),axis=0))
        raise ValueError
        print(self._stoich_mat * self._k_vec[:, np.newaxis])
        print(conc[np.newaxis, :] ** np.abs(self._stoich_mat))
        print(self._stoich_mat * self._k_vec[:, np.newaxis] * conc[np.newaxis, :] ** np.abs(self._stoich_mat))
        print(np.sum(self._stoich_mat * self._k_vec[:, np.newaxis] * conc[np.newaxis, :] ** np.abs(self._stoich_mat), axis=0))
        raise ValueError
        #k_conc_mat = self._k_vec * np.prod(np.power(conc, np.abs(self._stoich_mat)), axis=1
        print(self._stoich_mat)
        print(self._k_vec)
        print(self._stoich_mat*self._k_vec)
        print(conc[:, np.newaxis]**np.abs(self._stoich_mat))
        print(self._stoich_mat*self._k_vec*conc[:, np.newaxis]**np.abs(self._stoich_mat))
        print(np.sum(self._ode_mat.T*conc[:, np.newaxis]**np.abs(self._stoich_mat),axis=0))  # axis 0 is combine rows 
        print((self._stoich_mat @ self._k_vec.T) @ np.power(conc, np.abs(self._stoich_mat))) # fast

        rates = self._k_vec * np.power(conc, np.abs(self._stoich_mat))
        dcdt = np.sum(self._stoich_mat.T * rates, axis=1)
        print(rates)
        print(dcdt)

        raise ValueError
        return (self._stoich_mat @ self._k_vec.T) @ np.power(conc, np.abs(self._stoich_mat))
    

    def _construct_ode_mat(self):
        """Constructs the ODE matrix for the system's kinetics."""
        n_species = len(self.species)
        self._ode_mat = np.zeros((n_species, n_species))

        for tr in self.transitions.values():
            rate_constant = tr['value']
            for from_idx, coeff in tr['from_idx']:
                self._ode_mat[from_idx, from_idx] -= coeff * rate_constant
                for to_idx, coeff_to in tr['to_idx']:
                    self._ode_mat[to_idx, from_idx] += coeff * rate_constant

        self._ode_mat = self._ode_mat.T  # Transpose to have species as columns like concentration
    
    def simulate_deterministic_mat(self, t, method='BDF', rtol=1e-6, atol=1e-8, output_raw=False):
        """Simulates the system deterministically using the ODE matrix."""
        conc0 = np.array([np.atleast_1d(sp['conc'])[0] for _, sp in self.species.items()])
        self.log_dcdts()

        solution = solve_ivp(self._dcdt5, (t[0], t[-1]), conc0, method=method, t_eval=t, rtol=rtol, atol=atol)
        if not solution.success:
            raise RuntimeError("ODE solver failed: " + solution.message)

        # Update species concentrations
        self.t = t
        for name,data in self.species.items():
            data['conc'] = solution.y[data['index']]

        self.logger.info("ODEs solved successfully.")

        if output_raw:
            return solution
        else:
            return


    def _dcdt(self, t, concentrations):
        """
        Compute the derivative of concentrations with respect to time.
        Can directly handle non-linear reactions (eg stoich coeff > 1)

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
            rate = rate_constant * np.prod([concentrations[self.species[sp_name]['index']] ** coeff for sp_name,coeff in tr['from']])
            # Iterating through like this is beneficial because it captures stoichiometry that is evident in the list rather than coefficient 
                # (eg "from": ["E","E", "I"] is equal to "from": ["2E", "I"])
            for sp_name,coeff in tr['from']:
                dcdt_arr[self.species[sp_name]['index']] -= coeff * rate
            for sp_name,coeff in tr['to']:
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




