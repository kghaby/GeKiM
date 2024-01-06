# A + A' <--> B <--> C <--> D
# A + A' <----> X <----> D

from scipy.integrate import solve_ivp
import numpy as np
#from numba import njit
from scipy.linalg import eig

# Refactor the dcdt method into a standalone function to be compiled into machine code
#@njit
def dcdt_fourR(t, concArr, params):
    """
    Define the function that describes the rate of change of the concentrations
    """
    kon, koff, k1f, k1b, kqmf, kqmb = params
    L, P, L_P, LP  = concArr
    #Kd=(L*P)/(L__P+L_P+LP)
    dLdt = ((koff*k1b)/(k1f+k1b))*L_P - kon*L*P
    dPdt = ((koff*k1b)/(k1f+k1b))*L_P - kon*L*P #+ kqmf*L_P #last term is for testing substrate model where P becomes free after catalysis. This term prevents P from limiting LP production!!
    dL_Pdt = kon*L*P + kqmb*LP - (((koff*k1b)+(kqmf*k1f))/(k1f+k1b))*L_P
    dLPdt = ((kqmf*k1f)/(k1f+k1b))*L_P - kqmb*LP
    return [dLdt, dPdt, dL_Pdt, dLPdt]

class FourStateR():
    """
    L + P  <--kon/koff-->  L__P  <--k1f/k1b-->  L_P  <--kqmf/kqmb-->  LP

    L__P and L_P are grouped into one intermediate state. 
    """
    def __init__(self,Ki,koff,intOOM,Pf,kqmf,kqmb,conc0Arr):
        self.Ki = Ki
        self.koff = koff
        self.kon = self.koff / Ki
        self.intOOM = intOOM
        self.Pf = Pf
        self.Pb = 1-self.Pf
        self.k1f, self.k1b = self.getIntRates(self.intOOM, self.Pf)
        self.kqmf=kqmf
        self.kqmb=kqmb
        self.conc0Arr=conc0Arr #L,P,X,LP
        self.koff_eff = self.Pb * self.koff
        self.kqmf_eff = self.Pf * self.kqmf 
        self.ratioDict = {}

        # Eigenvalue analysis
        self.J = self.jacobian(self.conc0Arr)
        self.eigenvalues, self.eigenvectors = eig(self.J)
        #sort
        idx = self.eigenvalues.argsort()[::-1]   
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:,idx]
        
        negative_eigenvalues = [val for val in np.real(self.eigenvalues) if val < -1e-9]
        self.slowest_eigenvalue = max(negative_eigenvalues) if negative_eigenvalues else None

        
    def getIntRates(self,intOOM,Pf):
        k1f=Pf*10**(float(intOOM)+1)
        k1b=10**(float(intOOM)+1)-k1f
        return k1f,k1b

    
    def solve(self, t):
        t_span = (t[0], t[-1])
        params = (self.kon, self.koff, self.k1f, self.k1b, self.kqmf, self.kqmb)
        solution = solve_ivp(dcdt_fourR, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution.y.T
    

    def jacobian(self,concArr):
        '''
        concArr[0] should be ligand and concArr[1] should be protein   
        '''
        # Initialize the 4x4 Jacobian
        J = np.zeros((4, 4))

        # Derivatives for dL/dt
        J[0, 0] = -self.kon * concArr[1]  # dL/dL
        J[0, 1] = -self.kon * concArr[0]  # dL/dP
        J[0, 2] = (self.koff * self.k1b) / (self.k1f + self.k1b)  # dL/dL_P
        # Derivatives for dP/dt
        J[1, 0] = -self.kon * concArr[1]  # dP/dL
        J[1, 1] = -self.kon * concArr[0]  # dP/dP
        J[1, 2] = (self.koff * self.k1b) / (self.k1f + self.k1b) + self.kqmf  # dP/dL_P
        # Derivatives for dL_P/dt
        J[2, 0] = self.kon * concArr[1]  # dL_P/dL
        J[2, 1] = self.kon * concArr[0]  # dL_P/dP
        J[2, 2] = -(self.koff * self.k1b + self.kqmf * self.k1f) / (self.k1f + self.k1b)  # dL_P/dL_P
        J[2, 3] = self.kqmb  # dL_P/dLP
        # Derivatives for dLP/dt
        J[3, 2] = (self.kqmf * self.k1f) / (self.k1f + self.k1b)  # dLP/dL_P
        J[3, 3] = -self.kqmb  # dLP/dLP

        return J

