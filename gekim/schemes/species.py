from dataclasses import dataclass, field
from typing import Union, Optional
import numpy as np
from sympy import symbols, Symbol


@dataclass
class Species:
    """
    Initialize a species object.

    Parameters
    ----------
    name : str
        Name of the species.
    y0 : float or np.ndarray or int
        Initial concentration of the species.
        Array Example: `{"Ligand":np.linspace(1,1500,100)}` for a Michaelis-Menten ligand concentration scan.
    label : str, optional
        Useful for plotting. Will default to NAME.
    color : str, optional
        Useful for plotting. Best added by ..utils.Plotting.assign_colors_to_species().
    combination_rule : str, optional
        Determines how y0 values should be combined with others. Only relevant if the y0 is an array.
        'elementwise' means values will be combined elementwise with other species.
        'product' means the Cartesian product of y0 values will be taken with other species' y0 values.
    """
    name: str
    y0: Union[float, np.ndarray, int] = 0.0 
    label: Optional[str] = None
    color: Optional[str] = None
    combination_rule: str = 'elementwise'
    index: Optional[int] = field(init=False, default=None) 
    sym: Symbol = field(init=False)

    sim_kwargs: Optional[dict] = field(init=False, default_factory=dict)  # simulation kwargs, if any

    simin: Optional[dict] = field(init=False, default=None)  # simulation input, if any
    simout: Optional[dict] = field(init=False, default=None)  # simulation output, if any

    def __post_init__(self):
        
        if self.label is None:
            self.label = self.name

        self.sym = symbols(self.name)
        
        # Ensure y0 is stored as a numpy array
        if isinstance(self.y0, (int, float)):
            self.y0 = np.array([self.y0], dtype=float)
        elif isinstance(self.y0, np.ndarray):
            self.y0 = np.atleast_1d(self.y0).astype(float)
        else:
            raise TypeError(f"y0 must be a number or numpy array. Got {type(self.y0)}")
        # Validate combination_rule
        if self.combination_rule not in ('elementwise', 'product'):
            raise ValueError(f"Invalid combination_rule '{self.combination_rule}' for species '{self.name}'.")

    def __repr__(self):
        conc_display = self.y0.tolist() if isinstance(self.y0, np.ndarray) else self.y0
        return f"{self.name} (Initial: {conc_display}, Label: {self.label})"
