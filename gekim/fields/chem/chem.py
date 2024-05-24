import numpy as np

def arrhenius(A,Ea,T=300,kB=0.001987): #kcal/mol
    k = A * np.exp(-Ea/(kB*T))
    return k