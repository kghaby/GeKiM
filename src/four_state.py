# A + A' <--> B <--> C <--> D

from scipy.integrate import solve_ivp
import numpy as np
from numba import njit

# Refactor the dcdt method into a standalone function to be compiled into machine code
@njit
def dcdt_four(t, concArr, params):
    """
    Define the function that describes the rate of change of the concentrations
    """
    kon, koff, k1f, k1b, kqmf, kqmb = params
    L, P, L__P, L_P, LP  = concArr
    dLdt = koff*L__P - kon*L*P
    dPdt = koff*L__P - kon*L*P #+ kqmf*L_P #last term is for testing substrate model where P becomes free after catalysis. This term prevents P from limiting LP production!!
    dL__Pdt = kon*L*P - koff*L__P - k1f*L__P + k1b*L_P 
    dL_Pdt = k1f*L__P - k1b*L_P - kqmf*L_P + kqmb*LP
    dLPdt = kqmf*L_P - kqmb*LP
    return [dLdt, dPdt, dL__Pdt, dL_Pdt, dLPdt]

class FourState():
    """
    L + P  <--kon/koff-->  L__P  <--k1f/k1b-->  L_P  <--kqmf/kqmb-->  LP
    """
    def __init__(self,Ki,koff,intOOM,Pf,kqmf,kqmb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.intOOM=intOOM
        self.Pf=Pf
        self.k1f, self.k1b = self.getIntRates(self.intOOM, self.Pf)
        self.kqmf=kqmf
        self.kqmb=kqmb
        self.conc0Arr=conc0Arr
        self.ratioDict = {}
        
    def getIntRates(self,intOOM,Pf):
        k1f=Pf*10**(float(intOOM)+1)
        k1b=10**(float(intOOM)+1)-k1f
        return k1f,k1b

    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.k1f, self.k1b, self.kqmf, self.kqmb)
        solution = solve_ivp(dcdt_four, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution.y.T

    def survivalTime(t,sol):
        """
        Calc survival time of last state when kon = 0 
        """
        # Find the index of the time at which LP reaches 0.5
        LP = sol[:, 4]
        time_index = np.where(LP <= 0.5)[0][0]

        # The time at which LP reaches 0.5 is at the corresponding index in the time array
        time_to_reach_half = t[time_index]
        return time_to_reach_half
    
    def resTime(t,sol,LP0):
        """
        Calc survival time of bound state when kon = 0 
        """
        # Find the index of the time at which LP reaches 0.5
        boundL = sol[:, 2]+sol[:, 3]+sol[:, 4]
        try:
            time_index = np.where(boundL <= (LP0/2))[0][0]
        except:
            time_index=-1
        # The time at which LP reaches 0.5 is at the corresponding index in the time array
        time_to_reach_half = t[time_index]
        return time_to_reach_half

    @classmethod
    def koff_eff(cls, koff, Pf):
        """
        koff_eff (eff. off-rate constant) calculation.
        
        Args:
            koff: off-rate constant (s^-1)
            Pf: fraction of forward intermediate 
        """
        return koff * (1-Pf)
    
    @classmethod
    def Ki_eff(cls, kon, koff, Pf):
        """
        Ki_eff (effective eq. inhib. dissociation constant) calculation
        
        Args:
            kon: on-rate constant (nM^-1*s^-1)
            koff: off-rate constant (s^-1)
            Pf: fraction of forward intermediate 
        """
        return (koff * (1-Pf)) / kon
    
    @classmethod
    def KI_eff(cls, kon, koff, Pf, kinact):
        """
        KI (eff. inhibition constant) calculation
        
        Args:
            kon: on-rate constant (nM^-1*s^-1)
            koff: off-rate constant (s^-1)
            Pf: fraction of forward intermediate 
            kinact: inactivation (last irrev step) rate 
        """
        return (koff * (1-Pf) + kinact * Pf) / kon

    @classmethod
    def kinact_eff(cls, Pf, kinact):
        """
        kinact_eff (eff. inactivation rate constant) calculation.
        
        Args:
            Pf: fraction of forward intermediate 
            kinact: inactivation (last irrev step) rate 
        """
        return kinact*Pf
    
    @classmethod
    def specificity_eff(cls, kon, koff, Pf, kinact):
        """
        specificity_eff (eff. specificity constant) calculation (kinact_eff/KI_eff)
        """
        return cls.kinact_eff(Pf, kinact)/cls.KI_eff(kon, koff, Pf, kinact)
    