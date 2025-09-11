from dataclasses import dataclass, field
import re
from sympy import symbols, Symbol
from typing import Union, Optional

from ..utils.helpers import integerable_float

@dataclass
class Transition:
    """
    Parameters
    ----------
    name : str
        Name of the rate constant.
    k : float
        Value of the rate constant.
    source : list
        List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings.
    target : list
        List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings.
    label : str, optional
        Could be useful for plotting. Will default to NAME.
    """
    name: str
    k: float = None
    source: list[Union[str, tuple[Union[str, float], Union[str, float]]]] = field(default_factory=list)
    target: list[Union[str, tuple[Union[str, float], Union[str, float]]]] = field(default_factory=list)
    label: Optional[str] = None
    index: Optional[int] = field(init=False, default=None)
    sym: Symbol = field(init=False)
    linear: bool = field(init=False)

    sim_kwargs: Optional[dict] = field(init=False, default_factory=dict)  # simulation kwargs, if any

    simin: Optional[dict] = field(init=False, default=None)  # simulation input, if any
    simout: Optional[dict] = field(init=False, default=None)  # simulation output, if any

    def __post_init__(self):
        
        if self.label is None:
            self.label = self.name
            
        self.sym = symbols(self.name)
        
        # Format source and target into standardized (species_name, coeff) lists
        self.source = Transition._format_state(self.source, direction="source")
        self.target = Transition._format_state(self.target, direction="target")
        
        self.linear = self.is_linear()
        
    def __repr__(self):
        src_str = ' + '.join([f"{coeff}*{sp}" for sp, coeff in self.source]) or '∅'
        tgt_str = ' + '.join([f"{coeff}*{sp}" for sp, coeff in self.target]) or '∅'
        return f"{self.name} ({self.k}): {src_str} -> {tgt_str}"

    def is_linear(self):
        """
        Check if a transition is linear.

        Returns
        -------
        bool
            True if the transition is linear, False otherwise.

        Notes
        -----
        A first-order reaction must have exactly one source species (with a stoichiometric coefficient of 1).
        """
        return len(self.source) == 1 and self.source[0][1] == 1

    @staticmethod
    def _parse_species_string(species_str: str) -> tuple[str, Union[float, int]]:
        """
        Extract coefficient and species name from species string.

        Parameters
        ----------
        species_str : str
            A species string, e.g., '2A'.

        Returns
        -------
        tuple
            A tuple of species name (str) and stoichiometric coefficient (int|float).
        """
        match = re.match(r"(-?\d*\.?\d*)(\D.*)", species_str)
        if match and match.groups()[0]:
            coeff = match.groups()[0]
            if coeff == '-':
                coeff = -1
            coeff = integerable_float(float(coeff))
        else:
            coeff = 1
        name = match.groups()[1] if match else species_str
        return name, coeff

    @staticmethod
    def _format_state(state: Union[str, tuple, list], direction: Optional[str] = None) -> list[tuple[str, float]]:
        """
        Format a transition by extracting and combining coefficients and species names.
        Is idempotent.

        Parameters
        ----------
        state : list
            State descriptor. List of (SPECIES, COEFF) tuples or "{COEFF}{SPECIES}" strings.
        direction : str, optional
            Direction of the transition. Default is None. Can be "source" or "target".

        Returns
        -------
        list
            List of (SPECIES, COEFF) tuples.

        Raises
        ------
        ValueError
            If the transition or species tuples are invalid.
        """
        parsed_species = {}
        
        # Normalize to list
        state_list = []
        if state is None:
            state_list = []
        elif isinstance(state, list):
            state_list = state
        else:
            state_list = [state]
        
        # Parse each species in the state
        for sp in state_list:
            if isinstance(sp, str):
                name, coeff = Transition._parse_species_string(sp)
            elif isinstance(sp, tuple):
                if len(sp) == 2:
                    if isinstance(sp[0], str) and isinstance(sp[1], (int, float)):
                        name, coeff = sp
                    elif isinstance(sp[1], str) and isinstance(sp[2], (int, float)):
                        coeff, name = sp
                    else:
                        raise ValueError(f"Invalid species tuple '{sp}' in transition '{state}'.")
                else:
                    raise ValueError(f"Invalid species tuple '{sp}' in transition '{state}'.")
            else:
                raise ValueError(f"Invalid species '{sp}' in transition '{state}'.")
            if direction == "source" and coeff < 0:
                raise ValueError(f"Negative coefficient '{coeff}' in source of transition '{state}'.")
            if name in parsed_species:
                parsed_species[name] += coeff # combine coeffs
            else:
                parsed_species[name] = coeff
        state = [(name, coeff) for name, coeff in parsed_species.items()]
        return state
