from .simulator import Simulator
import numpy as np

class Gillespie(Simulator):
    """
    Gillespie's algorithm for stochastic simulation.
    """

    def simulate(self, t_max, conc0, num_replicates, output_times):
        results = [self._simulate_replicate(t_max, conc0, output_times) for _ in range(num_replicates)]
        return self._aggregate_replicate_data(results)
    
    #TODO: max iter for while loop
    def _simulate_replicate(self, t_max, conc0, output_times):
        times, states = [0], [conc0]
        while times[-1] < t_max:
            rates, transitions = self._calculate_transition_rates(states[-1])
            total_rate = np.sum(rates)
            if total_rate == 0: break
            time_step = np.random.exponential(1/total_rate)
            if (new_time := times[-1] + time_step) > t_max: break
            times.append(new_time)
            states.append(self._apply_transition(states[-1], transitions[np.random.choice(len(transitions), p=rates/total_rate)]))
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
