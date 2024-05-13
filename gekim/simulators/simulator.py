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
        rates = []
        for tr in self.scheme.transitions.values():
            rate = tr.k * np.prod([state[self.scheme.species[sp_name].index] ** coeff for sp_name, coeff in tr.source])
            rates.append(rate)
        return np.array(rates)

    def _apply_transition(self, current_state, transition):
        new_state = np.array(current_state)
        # Apply the transition
        for sp_name, coeff in transition.source:
            new_state[self.scheme.species[sp_name].index] -= coeff
        for sp_name, coeff in transition.target:
            new_state[self.scheme.species[sp_name].index] += coeff
        return new_state

