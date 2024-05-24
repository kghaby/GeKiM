import numpy as np
from .base_simulator import BaseSimulator

    #TODO: add an option called "continue" which takes an integer which points to the index of the run that its continuing from
    #TODO: precompile (repeatable) algo in c or fortran
    #TODO: S2.alt1 breaks this from negative rates somehow
    #TODO: running prob? and test more
    #TODO: type hints on algo
    #TODO: 2S I goes neg
    #TODO: warn if transition is not linear

class Gillespie(BaseSimulator):
    """
    Gillespie's algorithm for stochastic simulation.
    Does not work if any transitions are > (pseudo-)first order.  
    """
    def setup(self):
        pass

    def _process_simouts(self, simouts, y0_mat_len):
        if y0_mat_len == 1:
            simout = simouts[0]
            self.system.simout["t"] = simout['t']
            self.system.simout["prob_dist"] = simout['prob_dist']
            for sp_name, sp_data in self.system.species.items():
                sp_data.simout["prob_dist"] = simout['prob_dist'][:, sp_data.index]
        else:
            self.simout["t"] = [simout['t'] for simout in simouts]
            self.simout["prob_dist"] = [simout['prob_dist'] for simout in simouts]
            for sp_name, sp_data in self.system.species.items():
                sp_data.simout["prob_dist"] = [simout['prob_dist'][:, sp_data.index] for simout in simouts]
    
    def simulate(self, t_max, num_replicates=1, output_times=None, output_raw=False,**kwargs):
        """
        Run a Gillespie simulation on the system.
        
        Parameters:
        ----------
        t_max : float
            The maximum time to simulate.
        num_replicates : int, optional
            The number of replicates to run. Default is 1.
        output_times : array-like, optional
            The times at which to output the state of the system. Default is None.
        output_raw : bool, optional
            Whether to return the raw simulation output. Default is False.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the simulation.

        Returns:
        -------
        dict or list of dict
            The simulation output. If `output_raw` is True, returns a dictionary or a list of dictionaries containing the simulation output for each replicate. If `output_raw` is False, returns None.

        """
        y0_mat = self._make_y0_mat()
        y0_mat_len = len(y0_mat)
        raw_simouts = [self.simulate_single(t_max, y0, num_replicates, output_times,**kwargs) for y0 in y0_mat]
        self._process_simouts(raw_simouts, y0_mat_len)
        if output_raw:
            if y0_mat_len == 1:
                self.system.log.info("Returning raw simulation output.\n")
                return raw_simouts[0]
            self.system.log.info("Returning list of raw simulation outputs.\n")        
            return raw_simouts
        else:
            self.system.log.info("Not returning raw simulation output. Use output_raw=True to return raw data.\n")
            return

    def simulate_single(self, t_max, conc0, num_replicates, output_times, max_iter=1000):
        simouts = [self._simulate_replicate(t_max, conc0, output_times, max_iter) for _ in range(num_replicates)]
        return self._aggregate_replicate_data(simouts)

    def _simulate_replicate(self, t_max, conc0, output_times, max_iter):
        times, states = [0], [conc0]

        while times[-1] < t_max and len(times) < max_iter:
            rates = self._calculate_transition_rates(states[-1])
            total_rate = np.sum(rates)
            if total_rate == 0:
                break
            time_step = np.random.exponential(1 / total_rate)
            if (new_time := times[-1] + time_step) > t_max:
                break
            times.append(new_time)
            chosen_transition = np.random.choice(len(rates), p=rates / total_rate)
            transitions = list(self.system.transitions.values())
            states.append(self._apply_transition(states[-1], transitions[chosen_transition]))

        return {'t': np.array(times), 'state': np.array(states)}

    def _aggregate_replicate_data(self, replicates):
        t_all = np.concatenate([rep['t'] for rep in replicates])
        t_edges = np.unique(t_all)
        prob_dist = np.mean([self._collect_states_at_times(rep['t'], rep['state'], t_edges) for rep in replicates], axis=0)
        return {'t': t_edges, 'prob_dist': prob_dist}

    def _collect_states_at_times(self, times, states, output_times):
        idxs = np.searchsorted(times, output_times, side='right') - 1
        idxs[idxs < 0] = 0
        return states[idxs]

    def _calculate_transition_rates(self, state):
        rates = []
        for tr in self.system.transitions.values():
            rate = tr.k * np.prod([state[self.system.species[sp_name].index] ** coeff for sp_name, coeff in tr.source])
            rates.append(rate)
        return np.array(rates)

    def _apply_transition(self, current_state, transition):
        new_state = np.array(current_state)
        for sp_name, coeff in transition.source:
            new_state[self.system.species[sp_name].index] -= coeff
        for sp_name, coeff in transition.target:
            new_state[self.system.species[sp_name].index] += coeff
        return new_state