class Params:
    @staticmethod
    def KM(kon, koff, kcat):
        """
        KM (i.e. Michaelis-Menten constant, KA, Khalf, KD (not to be confused with Kd)) calculation.
        
        Parameters
        ----------
        kon : float
            On-rate constant (CONC^-1*TIME^-1)
        koff : float
            Off-rate constant (TIME^-1)
        kcat : float
            Irreversible catalysis rate constant
        
        Returns
        -------
        float
            The calculated Michaelis-Menten constant (KM)
        """
        return (koff + kcat) / kon