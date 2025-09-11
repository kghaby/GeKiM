from collections import OrderedDict
from typing import Union, Optional
from sympy import Symbol
from copy import deepcopy

from .species import Species
from .transition import Transition
from ..utils.plotting import assign_colors_to_species
from ..utils.logging import Logger

class Scheme:
    """
    Defines the network topology of a kinetic model (species and transitions) and contains the parameters.
    Can be built from a config dictionary or by adding species/transitions programmatically.
    """
    def __init__(self, config: Optional[dict] = None, name: Optional[str] = None,
                 color_kwargs: Optional[dict] = None, log: Logger = None, quiet: bool = False):
        self.name = name if name else "Unnamed"
        self.log = log if log else Logger(quiet=quiet)

        # Preserve insertion order for species and transitions
        self.species: OrderedDict[str, Species] = OrderedDict()
        self.transitions: OrderedDict[str, Transition] = OrderedDict()
        
        if config:
            self.config = deepcopy(config)
            self.load_from_dict(config)
            
        self.color_kwargs = color_kwargs if color_kwargs else {}
        # if self.num_species > 0:
        #     self._color_species()
            
    def __repr__(self):
        lines = [f"Scheme '{self.name}': {self.num_species} species, {self.num_transitions} transitions",
                 "Transitions:"]
        lines += [f"  {repr(tr)}" for tr in self.transitions.values()]
        return "\n".join(lines)
    
    def _color_species(self):
        """
        Assign colors to species based on the color_kwargs.
        Uses `gekim.utils.plotting.assign_colors_to_species()`.
        """
        if not self.species:
            raise ValueError("No species defined in the scheme.")
        
        assign_colors_to_species(self, **self.color_kwargs)

    def add_species(self, 
                    name: str, 
                    y0: Union[float, list, Symbol] = 0.0,
                    label: Optional[str] = None, 
                    color: Optional[str] = None,
                    combination_rule: str = 'elementwise'
                    ) -> Species:
        if name in self.species:
            raise ValueError(f"Species '{name}' already exists.")
        sp = Species(name=name, 
                     y0=y0, 
                     label=label, 
                     color=color,
                     combination_rule=combination_rule
                     )
        self.species[name] = sp
        # if color is None:
        #     self._color_species()
        self._reindex_species()
        self.log.info(f"Added species '{name}' with initial concentration {y0} at index {sp.index}.")
        return sp

    def remove_species(self, name: str):
        if name not in self.species:
            raise KeyError(f"Species '{name}' not found.")
        # Ensure no transitions involve this species
        involved = [tr_name for tr_name, tr in self.transitions.items()
                    if any(sp == name for sp, _ in tr.source) or any(sp == name for sp, _ in tr.target)]
        if involved:
            raise RuntimeError(f"Cannot remove species '{name}' because it is involved in transitions: {involved}")
        # Remove species
        del self.species[name]
        self._reindex_species()
        self.log.info(f"Removed species '{name}'.")

    def add_transition(self, 
                       name: str, 
                       k: Union[float, str, Symbol, None] = None,
                       source: Union[str, list, tuple] = None,
                       target: Union[str, list, tuple] = None,
                       label: Optional[str] = None
                       ) -> Transition:
        if name in self.transitions:
            raise ValueError(f"Transition '{name}' already exists.")
        tr = Transition(name=name, 
                        k=k, 
                        source=source or [], 
                        target=target or [],
                        label=label
                        )
        self.transitions[name] = tr
        self._reindex_transitions()
        self.log.info(f"Added transition '{name}' with rate {k} at index {tr.index}.")
        return tr

    def remove_transition(self, name: str):
        if name not in self.transitions:
            raise KeyError(f"Transition '{name}' not found.")
        del self.transitions[name]
        self._reindex_transitions()
        self.log.info(f"Removed transition '{name}'.")

    def load_from_dict(self, config: dict):
        if 'species' not in config or 'transitions' not in config:
            raise ValueError("Config must contain 'species' and 'transitions' keys.")
        
        # Load species
        for sp_name, data in config['species'].items():
            y0 = data.get('y0', 0.0)
            label = data.get('label')
            color = data.get('color')
            comb_rule = data.get('combination_rule', 'elementwise')
            self.add_species(sp_name, y0=y0, label=label, color=color,
                             combination_rule=comb_rule)
            
        # Load transitions
        for tr_name, data in config['transitions'].items():
            k = data.get('k', None)
            source = data.get('source', [])
            target = data.get('target', [])
            label = data.get('label')
            self.add_transition(tr_name, k=k, source=source, target=target,
                                 label=label)
            
        self._reindex_species()
        self._reindex_transitions()
        self._validate_species_labels()

    def _reindex_species(self):
        for idx, name in enumerate(self.species.keys()):
            self.species[name].index = idx

    def _reindex_transitions(self):
        for idx, name in enumerate(self.transitions.keys()):
            self.transitions[name].index = idx

    def _validate_species_labels(self):
        labels = set()
        for sp in self.species.values():
            if sp.label in labels:
                raise ValueError(f"Duplicate species label '{sp.label}' detected.")
            labels.add(sp.label)

    @property
    def num_species(self) -> int:
        return len(self.species)

    @property
    def num_transitions(self) -> int:
        return len(self.transitions)
