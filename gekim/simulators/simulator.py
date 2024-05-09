from abc import ABC, abstractmethod
import numpy as np

class Simulator(ABC):
    """
    Base class for all simulation algorithms.
    """

    def __init__(self, scheme):
        self.scheme = scheme

    @abstractmethod
    def simulate(self, t_max, conc0, num_replicates, output_times, **kwargs):
        """
        Abstract method to perform the simulation. Must be implemented by subclasses.
        """
        pass

    def _calculate_transition_rates(self, state):
        rates, transitions = [], []
        for tr_name, tr in self.scheme.transitions.items():
            rate = tr.k * np.prod([state[self.scheme.species[sp_name].index] ** coeff for sp_name, coeff in tr.source])
            rates.append(rate)
            transitions.append((tr.source, tr.target))
        return np.array(rates), transitions

    def _apply_transition(self, current_state, transition):
        new_state = np.array(current_state)
        for sp_name, coeff in transition[0]: new_state[self.scheme.species[sp_name].index] -= coeff
        for sp_name, coeff in transition[1]: new_state[self.scheme.species[sp_name].index] += coeff
        return new_state
