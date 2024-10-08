from abc import ABC, abstractmethod
import numpy as np
from itertools import product
import weakref
from ..schemes.n_state import NState

# TODO: make base output class

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
        num_species = len(self.system.species)

        elementwise_groups = []
        product_groups = []
        first_sp_name = None

        for sp_name, sp_data in self.system.species.items():
            sp_data.y0 = np.atleast_1d(np.atleast_1d(sp_data.y0).flatten())
            y0_array = sp_data.y0

            if sp_data.combination_rule == 'elementwise' and len(y0_array) > 1:
                if len(elementwise_groups) == 0:
                    first_sp_name = sp_name
                    elementwise_groups.append(y0_array[:, None])
                else:
                    # Zip 
                    if len(y0_array) != elementwise_groups[0].shape[0]:
                        raise ValueError(f"Mismatch in y0 lengths for elementwise combination: {len(y0_array)} ({sp_name}) vs {elementwise_groups[0].shape[0]} ({first_sp_name})")
                    elementwise_groups[-1] = np.hstack([elementwise_groups[-1], y0_array[:, None]])
            elif sp_data.combination_rule == 'product' or len(y0_array) == 1:
                product_groups.append((sp_data.index, y0_array))
            else:
                raise ValueError(f"Unknown combination rule '{sp_data.combination_rule}' for species '{sp_data.name}'.")

        if elementwise_groups:
            elementwise_mat = elementwise_groups[-1]
        else:
            elementwise_mat = np.ones((1, 1))  # Placeholder 

        # Combine elementwise and product combinations
        if product_groups:
            product_combinations = np.array(list(product(*(pg[1] for pg in product_groups))))
            if elementwise_mat.size > 1:
                combined_mat = np.array([np.hstack((e, p)) for e in elementwise_mat for p in product_combinations])
            else:
                combined_mat = product_combinations
        else:
            combined_mat = elementwise_mat

        # Correct columns based on the species index
        final_y0_mat = np.zeros((combined_mat.shape[0], num_species))
        elementwise_idx = 0
        product_idx = elementwise_mat.shape[1] if elementwise_mat.size > 1 else 0
        for sp_name, sp_data in self.system.species.items():
            if sp_data.combination_rule == 'elementwise' and len(sp_data.y0) > 1:
                final_y0_mat[:, sp_data.index] = combined_mat[:, elementwise_idx]
                elementwise_idx += 1
            elif sp_data.combination_rule == 'product' or len(sp_data.y0) == 1:
                final_y0_mat[:, sp_data.index] = combined_mat[:, product_idx]
                product_idx += 1

        return final_y0_mat




