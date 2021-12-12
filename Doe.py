from main import calc
import numpy as np
import matplotlib.pyplot as plt
from artap.operators import LHSGenerator

# dk boundaries
dk_min = 0.008
dk_max = 0.03
dk_stp = 0.005
dk_ini = (dk_min + dk_max)/2
 
#nk boundaries
dn_min = 20
dn_max = 60
dn_stp = 5
dn_ini = (dn_min + dn_max)/2

# DOE setup
N = 15

parameters = [{'name': 'nk', 'initial_value': dn_ini, 'bounds': [dn_min, dn_max]},
              {'name': 'dk', 'initial_value': dk_ini, 'bounds': [dk_min, dk_max]}]

def doe_lhs():
    gen = LHSGenerator(parameters=parameters)
    gen.init(number=N)
    individuals = gen.generate()

    res_dP = []
    res_Tav= []
    res_nk = []
    res_dk = []

    for individual in individuals:
        par_nk = individual.vector[0]
        par_dk = individual.vector[1]
        res_nk.append(par_nk)
        res_dk.append(par_dk)
        
        [dP, Tav] = calc(par_nk,par_dk)
        res_dP.append(dP)
        res_Tav.append(Tav)
        
    return [res_nk, res_dk, res_dP, res_Tav]
