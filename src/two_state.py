# A <---> B

from scipy.integrate import odeint
import numpy as np

class TwoState():
    """
    L + P  <--kon/koff-->  L_P 
    """
    def __init__(self,Ki,koff,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.conc0Arr = conc0Arr
        self.sol=None

 
    def dcdt(self,concArr, t,kon,koff):
        """
        Define the function that describes the rate of change of the concentrations
        """
        L, P, L_P  = concArr
        dLdt = koff*L_P - kon*L*P
        dPdt = koff*L_P - kon*L*P 
        dL_Pdt = kon*L*P - koff*L_P
        return [dLdt, dPdt, dL_Pdt]

    def solve(self,t):
        def dcdt_wrapper(concArr, t):
            return self.dcdt(concArr, t, self.kon, self.koff)
        return odeint(dcdt_wrapper, self.conc0Arr, t)

    def survivalTime(t,sol,initialConc):
        """
        Calc survival time of last state when kon = 0 and initial conc of last state = 1
        """
        # Find the index of the time at which LP reaches 0.5
        L_P = sol[:, 2]
        time_index = np.where(L_P <= (initialConc/2))[0][0]

        # The time at which LP reaches 0.5 is at the corresponding index in the time array
        time_to_reach_half = t[time_index]
        return time_to_reach_half
    
    def resTime(t,sol,initialConc):
        """
        Calc residence time of bound states when kon = 0 
        """
        # Find the index of the time at which LP reaches 0.5
        boundL = sol[:, 2]
        try:
            time_index = np.where(boundL <= (initialConc/2))[0][0]
        except:
            time_index=-1
        # The time at which LP reaches 0.5 is at the corresponding index in the time array
        time_to_reach_half = t[time_index]
        return time_to_reach_half

