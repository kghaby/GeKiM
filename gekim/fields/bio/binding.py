
def res_time():
    raise NotImplementedError()

class Parameters:
    """
    Common place for parameters found in general binding kinetics literature.
    """
    @staticmethod
    def Kd(kon, koff):
        """
        Kd (i.e. dissociation constant) calculation
        
        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1)
        koff : float
            Off-rate constant (TIME^-1)
        
        Returns
        -------
        float
            The calculated dissociation constant (Kd)
        """
        return koff / kon
    
    @staticmethod
    def Keq(kon, koff):
        """
        Keq (i.e. equilibrium constant) calculation
        
        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1)
        koff : float
            Off-rate constant (TIME^-1)
        
        Returns
        -------
        float
            The calculated equilibrium constant (Keq)
        """
        return kon / koff

    
