import numpy as np
from scipy.integrate import solve_ivp
import re
import copy
#TODO: add error handling, documentation and logging info, and show the ODEs. Add stochastic method
class NState:
    def __init__(self, config):
        """
        Species can contain name, conc, label. Conc will be the initial concentration.
        Transitions can contain name, from, to, value, label.
        """
        self.config=copy.deepcopy(config)
        self.species = self.config['species']
        self.species_order = {name: idx for idx, name in enumerate(self.config['species'])}
        self.transitions = self.config['transitions']
        # Preprocess the transitions with coefficients and names
        for _, tr in self.transitions.items():
            tr['from'] = [self.identify_coeff(s) for s in tr['from']]
            tr['to'] = [self.identify_coeff(s) for s in tr['to']]
        self.sol = None

    @staticmethod
    def identify_coeff(species_str):
        # Extract coefficient and species name from species string
        match = re.match(r"(\d*)(\D.*)", species_str)
        coeff = int(match.groups()[0]) if match and match.groups()[0] else 1
        name = match.groups()[1] if match else species_str
        return coeff, name  # Return a tuple of coefficient and species name

    def dcdt(self, t, concentrations):
        dcdt_arr = np.zeros(len(self.species))
        for name, tr in self.transitions.items():
            rate_constant = tr['value']

            rate = rate_constant * np.prod([concentrations[self.species_order[name]] ** coeff for coeff, name in tr['from']])
            for coeff, name in tr['from']:
                dcdt_arr[self.species_order[name]] -= coeff * rate

            for coeff, name in tr['to']:
                dcdt_arr[self.species_order[name]] += coeff * rate

        return dcdt_arr

    def solve_ode(self, t):
        conc0 = np.array([sp['conc'] for _, sp in self.species.items()])
        t_span = (t[0], t[-1])
        solution = solve_ivp(
            lambda t, conc: self.dcdt(t, conc),
            t_span, conc0, t_eval=t, method='BDF', rtol=1e-6, atol=1e-8
        )
        self.sol = solution.y.T

    def gillespie_sim(self):
        pass  # Gillespie algorithm to be implemented