#%%

import numpy as np
from qutip import * 
# import qutip as q
import matplotlib.pyplot as plt
from Basis import *
N = 40

# %%
psi_inicial_i = q.basis(N,10) 
evolucion_temporal_estado_i, tiempo_i, tasa_inversion_i = Hamiltonian_2levels(psi_inicial_i, N , 300)

wignerslist_i = []
xvec = np.linspace(-10,10,300)

Wignersf = WignerEvolution(evolucion_temporal_estado_i, xvec, wignerslist_i)

# %%
AnimatedWigner(Wignersf,xvec,'wignerfock')
AnimatePopulation(tasa_inversion_i,tiempo_i,'populationwignerfock')

# %%
# Entropyf = []
# for i in range(len(evolucion_temporal_estado_Fock)):
#     Entropyf.append(entropy_vn(evolucion_temporal_estado_Fock[i]))

# fig, ax = plt.subplots(figsize=(6, 6))    
# ax.plot(tiempo,Entropyf , color = 'firebrick')
# ax.set_xlabel('Tiempo')
# ax.set_ylabel('Entropia')
# ax.set_title('Entropia de Von Neumann')

Fidelityfock= []
for i in range(len(evolucion_temporal_estado_i)):
    Fidelityfock.append(q.fidelity(evolucion_temporal_estado_i[i], q.fock(2*N,10)))

fig, ax = plt.subplots(figsize=(6, 6))    
ax.plot(tiempo_i,Fidelityfock , color = 'firebrick')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Fidelity')
ax.set_title('Fidelidad Fock')



# %%
psi_inicial_cat = (coherent(N, -3.0) + coherent(N, 3.0)) / np.sqrt(2)
evolucion_temporal_estado_cat, tiempo, tasa_inversionc = Hamiltonian_2levels(psi_inicial_cat, N , 300)
wignerslistc = []
xvec = np.linspace(-10,10,300)
Wignersc = WignerEvolution(evolucion_temporal_estado_cat, xvec, wignerslistc)
#%%
AnimatedWigner(Wignersc,xvec,'wignercat')
AnimatePopulation(tasa_inversionc,tiempo,'populationwignercat')
# %%
# Entropyc = []
# for i in range(len(evolucion_temporal_estado_cat)):
#     Entropyc.append(entropy_vn(evolucion_temporal_estado_cat[i]))

# fig, ax = plt.subplots(figsize=(6, 6))    
# ax.plot(tiempo,Entropyc , color = 'firebrick')
# ax.set_xlabel('Tiempo')
# ax.set_ylabel('Entropia')
# ax.set_title('Entropia de Von Neumann')

Fidelityc= []
for i in range(len(evolucion_temporal_estado_cat)):
    Fidelityc.appendf(q.fidelity(evolucion_temporal_estado_cat[i]))

fig, ax = plt.subplots(figsize=(6, 6))    
ax.plot(tiempo,Fidelityc , color = 'firebrick')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Fidelidad')
ax.set_title('Fidelidad cat state')


# %%
psi_inicial_Sq =  q.squeeze(N, 1.0)* q.basis(N,0) 
evolucion_temporal_estado_Sq, tiempo, tasa_inversionsq = Hamiltonian_2levels(psi_inicial_Sq, N , 300)

wignerslistsq = []
xvec = np.linspace(-10,10,300)

Wignerssq = WignerEvolution(evolucion_temporal_estado_Sq, xvec, wignerslistsq)
# %%
AnimatedWigner(Wignerssq,xvec,'wignersq')
AnimatePopulation(tasa_inversionsq,tiempo,'populationwignersq')
# %%
Entropysq = []
for i in range(len(evolucion_temporal_estado_Sq)):
    Entropysq.append(entropy_vn(evolucion_temporal_estado_Sq[i]))

fig, ax = plt.subplots(figsize=(6, 6))    
ax.plot(tiempo,Entropysq , color = 'firebrick')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Entropia')
ax.set_title('Entropia de Von Neumann')


# %%
psi_inicial_Sq_c =  (q.displace(N,-1)*q.squeeze(N, 1.0)*q.basis(N,0) + q.displace(N,1)*q.squeeze(N, 1.0)*q.basis(N,0)).unit()
evolucion_temporal_estado_Sq_c, tiempo, tasa_inversionsq_c = Hamiltonian_2levels(psi_inicial_Sq_c, N , 300)
wignerslistsq_c = []
xvec = np.linspace(-10,10,300)
Wignerssq_c = WignerEvolution(evolucion_temporal_estado_Sq_c, xvec, wignerslistsq_c)
# %%
AnimatedWigner(Wignerssq_c,xvec,'wignersq_c')
AnimatePopulation(tasa_inversionsq_c,tiempo,'populationwignersq_c')

# %%
Entropysq_c = []
for i in range(len(evolucion_temporal_estado_Sq_c)):
    Entropysq_c.append(entropy_vn(evolucion_temporal_estado_Sq_c[i]))

fig, ax = plt.subplots(figsize=(6, 6))    
ax.plot(tiempo,Entropysq_c , color = 'firebrick')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Entropia')
ax.set_title('Entropia de Von Neumann')

# %%


# %%
