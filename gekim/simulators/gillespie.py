import numpy as np
from .simulator import Simulator

class Gillespie(Simulator):
    """
    Gillespie's algorithm for stochastic simulation. Handles non-linear kinetics and complex stoichiometry.
    Does not work if any transitions are > (pseudo-)first order.  
    """

    def simulate(self, t_max, conc0, num_replicates, output_times, max_iter=1000):
        results = [self._simulate_replicate(t_max, conc0, output_times, max_iter) for _ in range(num_replicates)]
        return self._aggregate_replicate_data(results)

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
            transitions = list(self.scheme.transitions.values())
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

