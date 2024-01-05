# A + A' <----> B <----> C

from scipy.integrate import solve_ivp
import numpy as np
from numba import njit

# Refactor the dcdt method into a standalone function to be compiled into machine code
@njit    
def dcdt_three(t,concArr, params):
    """
    Define the function that describes the rate of change of the concentrations
    """
    kon,koff,kinactf,kinactb=params
    L, P, L_P, LP  = concArr
    dLdt = koff*L_P - kon*L*P
    dPdt = koff*L_P - kon*L*P 
    dL_Pdt = kon*L*P - koff*L_P - kinactf*L_P + kinactb*LP
    dLPdt = kinactf*L_P - kinactb*LP
    return [dLdt, dPdt, dL_Pdt, dLPdt]


class ThreeState():
    """
    L + P  <--kon/koff-->  L_P  <--kqmf/kqmb-->  LP
    """
    def __init__(self,Ki,koff,kinactf,kinactb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.kinactf=kinactf
        self.kinactb=kinactb
        self.conc0Arr = conc0Arr
        self.sol=None
        
    #solve_ivp version
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.kinactf, self.kinactb)
        solution = solve_ivp(dcdt_three, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution.y.T

    def survivalTime(t,sol,initialConc):
        """
        Calc survival time of last state when kon = 0 and initial conc of last state = 1
        """
        # Find the index of the time at which LP reaches 0.5
        LP = sol[:, 3]
        time_index = np.where(LP <= (initialConc/2))[0][0]

        # The time at which LP reaches 0.5 is at the corresponding index in the time array
        time_to_reach_half = t[time_index]
        return time_to_reach_half
    
    def resTime(t,sol,initialConc):
        """
        Calc residence time of bound states when kon = 0 
        """
        # Find the index of the time at which LP reaches 0.5
        boundL = sol[:, 2]+sol[:, 3]
        try:
            time_index = np.where(boundL <= (initialConc/2))[0][0]
        except:
            time_index=-1
        # The time at which LP reaches 0.5 is at the corresponding index in the time array
        time_to_reach_half = t[time_index]
        return time_to_reach_half
    
    #@classmethod
    #def Ki(cls, kon, koff):
    #    """
    #    Ki (eq. inhib. dissociation constant) calculation
    #    
    #    Args:
    #        kon: on-rate constant (nM^-1*s^-1)
    #        koff: off-rate constant (s^-1)
    #    """
    #    return koff / kon

    @classmethod
    def KI(cls, kon, koff, kinact):
        """
        KI (inhibition constant) calculation
        
        Args:
            kon: on-rate constant (nM^-1*s^-1)
            koff: off-rate constant (s^-1) 
            kinact: inactivation (last irrev step) rate 
        """
        return (koff + kinact) / kon
    
    @classmethod
    def specificity(cls, kon, koff, kinact):
        """
        specificity (specificity constant) calculation (kinact/KI)
        """
        return kinact/cls.KI(kon, koff, kinact)
    

