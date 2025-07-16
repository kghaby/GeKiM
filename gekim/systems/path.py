from dataclasses import dataclass, field
class Path:
    """
    Represents a path in a network of species transitions. 
    Is created by `NState.find_paths()`

    Attributes
    ----------
    species_path : list
        List of species objects representing the path.
    transitions_path : list
        List of transition objects representing the transitions along the path.
    probability : float
        Probability of the path relative to other paths from `species[0]` to `species[-1]`
    length : int
        Length of species_path.

    Methods
    -------
    __repr__()
        Returns a string representation of the species path.

    """

    def __init__(self, species_path, transitions_path, probability):
        self.species_path = species_path
        self.transitions_path = transitions_path
        self.probability = probability
        self.length = len(species_path)

    def __repr__(self):
        """
        Returns a string representation of the Path object.

        Returns
        -------
        str
            String representation of the Path object.

        """
        path_str = ' -> '.join(['+'.join([sp.name for sp in group]) if isinstance(group, list) else group.name for group in self.species_path])
        prob_fmt = "{:.2e}".format(self.probability)
        return f"Path(Length: {str(self.length).rjust(3)},\tProbability: {prob_fmt},\t{path_str})"