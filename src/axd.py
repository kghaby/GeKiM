# A <--> X <--> D
# X=B+C from A <--> B <--> C <--> D

from scipy.integrate import solve_ivp
import numpy as np
from numba import njit
from scipy.linalg import eig

# Refactor the dcdt method into a standalone function to be compiled into machine code
@njit
def dcdt_AXD(t, concArr, params):
    """
    Define the function that describes the rate of change of the concentrations
    """
    kon, koff, k1f, k1b, kqmf, kqmb = params
    A,X,D  = concArr
    dAdt = ((koff*k1b)/(k1f+k1b))*X - kon*A
    dXdt = kon*A + kqmb*D - (((koff*k1b)+(kqmf*k1f))/(k1f+k1b))*X
    dDdt = ((kqmf*k1f)/(k1f+k1b))*X - kqmb*D
    return [dAdt, dXdt, dDdt]

class AXD():
    """
    A  <--kon/koff-->  B  <--k1f/k1b-->  C  <--kqmf/kqmb-->  D

    B and C are grouped into one intermediate state X
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
        self.conc0Arr=conc0Arr #A,B,C,D
        self.koff_eff = self.Pb * self.koff
        self.kqmf_eff = self.Pf * self.kqmf 
        self.ratioDict = {}

        # Eigenvalue analysis
        self.J = self.jacobian()
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
        solution = solve_ivp(dcdt_AXD, t_span, self.conc0Arr, t_eval=t, args=(params,), method='BDF', rtol=1e-6, atol=1e-8)
        return solution.y.T
    

    def jacobian(self):
        J = np.array([
            [-self.kon, self.koff_eff, 0],
            [self.kon, -(self.koff_eff + self.kqmf_eff), self.kqmb],
            [0, self.kqmf_eff, -self.kqmb]
        ])
        return J

