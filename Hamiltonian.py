import numpy as np
from qutip import * 
import qutip as q
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
import matplotlib.cm as cm
import datetime as dt
from Basis import *


def Hamiltonian_2levels(psi_inic, N , t = 300, w0 = 1. , w = 1., lamda = 1):
    #Constantes del sistema
    # w0 = 1
    # w = 1
    # lamda = 1
    #Creamos los operadores para el hamiltoniano
    sig_atomo = q.tensor(q.destroy(2),q.identity(N)) #producto tensorial entre sigma 2 y la identidad N
    a_campo = q.tensor(q.identity(2),q.destroy(N)) #producto tensorial entre la identidad 2 y operador destruccion N
    sig_z = q.tensor(q.sigmaz(),q.identity(N)) #producto tensorial entre sigma z y la identidad N
    #Creamos el hamiltoniano para la evolución temporal
    H_int = 0.5 * w0 * sig_z + w*a_campo.dag()*a_campo + lamda*(sig_atomo.dag()*a_campo + sig_atomo*a_campo.dag())
    
    #estado basal de 2 niveles
    g_atomo = q.basis(2,0)#ground state
    e_atomo = q.basis(2,1)#excited state
    estado_e = q.tensor(e_atomo,psi_inic)
    estado_g = q.tensor(g_atomo,psi_inic)
    
    tiempo = np.linspace(0,25*lamda/w,t)
    #Dado el estado inicial estado_g, la evolución se calcula usando mesolve
    estado_final = q.mesolve(H_int,estado_g,tiempo)
    evolucion_temporal_estado = estado_final.states
    tasa_inversion = q.expect(sig_z, evolucion_temporal_estado)
    
    return evolucion_temporal_estado, tiempo, tasa_inversion