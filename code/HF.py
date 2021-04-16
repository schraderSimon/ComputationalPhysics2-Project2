import numpy as np
import matplotlib.pyplot as plt
from quantum_systems import ODQD, GeneralOrbitalSystem
import scipy
def construct_Density_matrix(C,number_electrons,l):
    n=np.eye(2*l)
    for i in range(number_electrons,2*l):
        n[i,i]=0
    P=C@n@C.conjugate().T
    return P

    P=np.zeros((2*l,2*l),dtype=np.complex64) # P matrix
    for tau in range(2*l):
        for sigma in range(2*l):
            for i in range(number_electrons):
                P[tau,sigma]=np.conj(C[sigma,i])*C[tau,i]
    return P
def costruct_Fock_matrix(P,l,number_electrons,system):
    udirect=np.zeros((2*l,2*l),dtype=np.complex64)
    uexchange=np.zeros((2*l,2*l),dtype=np.complex64)
    for tau in range(2*l):
        for sigma in range(2*l):
            for mu in range(2*l):
                for nu in range(2*l):
                    udirect[mu,nu]+=P[tau,sigma]*system.u[mu,sigma,nu,tau]
                    #uexchange[mu,nu]+=P[tau,sigma]*system.u[mu,sigma,tau,nu]
    F=system.h+udirect-uexchange
    return F


l=10
grids_length=10
num_grid_points=2001
omega=0.25
a=1e10
odho = ODQD(l, grids_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))
number_electrons=6
system=GeneralOrbitalSystem(n=number_electrons,basis_set=odho,anti_symmetrize=True)
print(system.u)

C=np.eye(2*l) #Create an initial guess for the coefficients
P=np.zeros((2*l,2*l))
for i in range(50):
    P_old=P
    P=construct_Density_matrix(C,number_electrons,l)
    if(20<i<30):
        P=(P_old+P)/2
    #print(P)
    F=costruct_Fock_matrix(P,l,number_electrons,system)
    #F=system.construct_fock_matrix(system.h,system.u)
    #print(F)
    epsilon, C = scipy.linalg.eigh(F)
    print(epsilon)
    #print(C[0,:])
    #print(np.sum(C[:,1]*C[:,1]))
