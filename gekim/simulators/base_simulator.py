from abc import ABC, abstractmethod
import numpy as np
from itertools import product
import weakref
from ..schemes.n_state import NState

class BaseSimulator(ABC):
    def __init__(self, system: NState):
        """
        Initialize the simulator.

        Parameters
        ----------
        system : NState
            The system object. 

        Notes
        -----
        The input NState instance, `system`, will be modified directly 
            by the simulator, whether the simulator is added as an attribute to `system` or not.
            
        Initializes the `simin` and `simout` dictionaries of system, species, and transitions objects.
        """
        system.log.info(f"Setting up solver {self.__class__}...\n")
        self.system = weakref.proxy(system)

        self.system.simin = {}
        self.system.simout = {}
        for _, sp_data in self.system.species.items():
            sp_data.simin = {}
            sp_data.simout = {}
        for _, tr_data in self.system.transitions.items():
            tr_data.simin = {}
            tr_data.simout = {}

        self.setup()

    @abstractmethod
    def setup(self):
        """
        Perform any necessary setup specific to the simulator.
        This method should be overridden by subclasses.
        """
        pass

    @abstractmethod
    def simulate(self, output_raw=False, **kwargs):
        """
        Run the simulation.
        This method should be overridden by subclasses.
        """
        pass

    def _make_y0_mat(self):
        """
        Create a matrix of initial conditions for the system.
        """
        combinations = product(*(
            np.atleast_1d(np.atleast_1d(sp_data.y0).flatten())
            for _, sp_data in self.system.species.items()
        ))
        y0_mat = np.vstack([comb for comb in combinations])
        return y0_mat



