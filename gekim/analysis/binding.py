
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
        
        Args:
            kon: on-rate constant (nM^-1*s^-1)
            koff: off-rate constant (s^-1)
        """
        return koff / kon
    
    @staticmethod
    def Keq(kon, koff):
        """
        Keq (i.e. equilibrium constant) calculation
        
        Args:
            kon: on-rate constant (nM^-1*s^-1)
            koff: off-rate constant (s^-1)
        """
        return kon / koff

    @staticmethod
    def KM(kon, koff, kcat):
        """
        KM (i.e. Michaelis-Menten constant, KA, Khalf, KD) calculation.
        
        Args:
            kon: on-rate constant (nM^-1*s^-1)
            koff: off-rate constant (s^-1) 
            kcat: irrev catalysis rate constant
        """
        return (koff + kcat) / kon
    
