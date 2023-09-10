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






def Hamiltonian3LevelsLadderOM(psi_inic, N, t, chi, eta, omega, omega_1, omega_2, omega_3):
    #we build the hamiltonian for a 3 level system in ladder configuration
    #this means that we will have transitions between levels, excited by an external field of a single mode
    omega_12 = omega_1 - omega_2
    omega_23 = omega_2 - omega_3
    Delta_l = np.abs(omega_12) - omega
    Delta_r = np.abs(omega_23) - omega
    
    a_field = q.tensor(q.qeye(3),q.destroy(N))
    
    # Create atomic operators for the three-level system
    b1_electron = q.tensor(q.destroy(3), q.qeye(N))
    b2_electron = q.tensor(q.qeye(3), q.destroy(N))
    b3_electron = q.tensor(q.qeye(3), q.destroy(N))
    
    #ESTOS NO ESTAN BUENOS!!!1 Pero me gustaría definirlo de esta manera más adelante
    
    # sigma11 = q.tensor(q.basis(3,0)*q.basis(3,0).dag(),q.identity(N)) 
    # sigma12 = q.tensor(q.basis(3,0)*q.basis(3,1).dag(),q.identity(N))
    # sigma21 = q.tensor(q.basis(3,1)*q.basis(3,0).dag(),q.identity(N))
    # sigma22 = q.tensor(q.basis(3,1)*q.basis(3,1).dag(),q.identity(N))
    # sigma23 = q.tensor(q.basis(3,1)*q.basis(3,2).dag(),q.identity(N))
    # sigma32 = q.tensor(q.basis(3,2)*q.basis(3,1).dag(),q.identity(N))
    # sigma33 = q.tensor(q.basis(3,2)*q.basis(3,2).dag(),q.identity(N))
    
    
    
    H_p = chi* a_field * b2_electron.dag() * b1_electron + eta * a_field * b3_electron.dag() * b2_electron + np.conj(chi)* a_field.dag() * b2_electron * b1_electron.dag() + np.conj(eta) * a_field.dag() * b3_electron * b2_electron.dag()
    
    P_e = b1_electron.dag() * b1_electron + b2_electron.dag() * b2_electron + b3_electron.dag() * b3_electron
    
    N_hat = a_field.dag() * a_field + b3_electron.dag() * b3_electron - b1_electron.dag() * b1_electron + q.eye(N)
    
    H_i = omega * (N_hat - 1) + (omega_2 - omega) * P_e
    
    H_ii = - Delta_l * b1_electron.dag() * b1_electron + Delta_r * b2_electron.dag() * b2_electron
    
    H = H_p + H_i + H_ii
    
    return H
    
    
def Hamiltonian3LevelsLadderTM(psi_inic, N, t, chi, eta,omega_r, omega_l, omega_1, omega_2, omega_3):
    #chi and eta are coupling constants (not necesarilly real)
    #I do not know when or how to define them yet!
    omega_12 = omega_1 - omega_2
    omega_23 = omega_2 - omega_3
    Delta_l = np.abs(omega_12) - omega_l
    Delta_r = np.abs(omega_23) - omega_r
    
    #Fiel operators , interact with:
    a_field_l = q.tensor(q.identity(3),q.destroy(N)) # levels 1 and 2
    a_field_r = q.tensor(q.identity(3),q.destroy(N)) # levels 2 and 3
    
    # Create atomic operators for the three-level system
    b1_electron = q.tensor(q.destroy(3), q.qeye(N))
    b2_electron = q.tensor(q.qeye(3), q.destroy(N))
    b3_electron = q.tensor(q.qeye(3), q.destroy(N))

    # Create Atom Hamiltonian
    # H_A = omega_1 * b1_electron.dag() * b1_electron + omega_2 * b2_electron.dag() * b2_electron + omega_3 * b3_electron.dag() * b3_electron
    
    # H_F = omega_l * a_field_l.dag()*a_field_l + omega_r*a_field_r.dag()*a_field_r

    H_p = chi* a_field_l * b2_electron.dag() * b1_electron + eta * a_field_r * b3_electron.dag() * b2_electron + np.conj(chi)* a_field_l.dag() * b2_electron * b1_electron.dag() + np.conj(eta) * a_field_r.dag() * b3_electron * b2_electron.dag()
    
    P_e = b1_electron.dag() * b1_electron + b2_electron.dag() * b2_electron + b3_electron.dag() * b3_electron
    
    N_hat_l = a_field_l.dag() * a_field_l - b1_electron.dag() * b1_electron + q.eye(N)
    
    N_hat_r = a_field_r.dag() * a_field_r + b3_electron.dag() * b3_electron
    
    H_i = omega_l*N_hat_r + omega_r*N_hat_l + (omega_2) * P_e - omega_l*q.eye(N)
    
    H_ii = - Delta_l * b1_electron.dag() * b1_electron + Delta_r * b3_electron.dag() * b3_electron + H_p
    
    H = H_i + H_ii
    
    return H
