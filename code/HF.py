import numpy as np
import numba
import matplotlib.pyplot as plt
from quantum_systems import ODQD, GeneralOrbitalSystem
import scipy
def construct_Density_matrix(C,number_electrons,l):
    """
    n=np.eye(2*l)
    for i in range(number_electrons,2*l):
        n[i,i]=0
    P=C@n@C.conjugate().T
    """
    slicy=slice(0,number_electrons)
    P=np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy])
    return P

def costruct_Fock_matrix(P,l,number_electrons,system,anti_symmetrize=True):

    """
    udirect=np.zeros((2*l,2*l),dtype=np.complex64)
    uexchange=np.zeros((2*l,2*l),dtype=np.complex64)
    for tau in range(2*l):
        for sigma in range(2*l):
            for mu in range(2*l):
                for nu in range(2*l):
                    udirect[mu,nu]+=P[tau,sigma]*system.u[mu,sigma,nu,tau]
                    if not anti_symmetrize:
                        uexchange[mu,nu]+=P[tau,sigma]*system.u[mu,sigma,tau,nu]
    """
    udirect=np.einsum("ts,msnt->mn",P,system.u)
    uexchange=0
    if not anti_symmetrize:
        uexchange=np.einsum("ts,mstn->mn",P,system.u)

    F=system.h+udirect-uexchange

    return F
def solve(system,number_electrons,number_basissets,C=None,anti_symmetrize=True,tol=1e-8,maxiter=100):
    if C is None:
        C=np.random.rand(2*number_basissets,2*number_basissets)
    P=np.zeros(C.shape)
    converged=False
    for i in range(maxiter):
        if(i>=1):
            convergence_difference=np.max(np.abs(P-P_old))
            print(convergence_difference)
            if (convergence_difference<tol):
                converged=True
                break
        P_old=P
        P=construct_Density_matrix(C,number_electrons,number_basissets)
        F=costruct_Fock_matrix(P,number_basissets,number_electrons,system,anti_symmetrize)
        epsilon, C = scipy.linalg.eigh(F)
        #print(C[0:4,0:4])
    return F,epsilon, C, converged


"""Ignore this, I was just playing around"""
def solve_DIIS(system,number_electrons,number_basissets,C=None,anti_symmetrize=True,tol=1e-8,maxiter=100):
    if C is None:
        C=np.random.rand(2*number_basissets,2*number_basissets)
    P=np.zeros(C.shape)
    Ps=np.zeros((P.shape[0],P.shape[1],5),dtype=np.complex64)
    converged=False
    for i in range(maxiter):
        if(i>=1):
            convergence_difference=np.max(np.abs(P-P_old))
            print(convergence_difference)
            if (convergence_difference<tol):
                converged=True
                break
        P_old=P
        P=construct_Density_matrix(C,number_electrons,number_basissets)
        P_con=P
        for k in range(4):
            Ps[:,:,k]=Ps[:,:,k+1]
        Ps[:,:,-1]=P

        if(i>=5):
            error_matrices=np.diff(Ps)
            B=np.zeros((5,5))
            for k in range(5):
                B[:,4]=-1
                B[4,:]=-1
            B[4,4]=0
            for k in range(4):
                for l in range(4):
                    B[k,l]=np.max(abs(np.dot(error_matrices[:,:,k].flatten(),error_matrices[:,:,l].flatten())))#,ord=inf)
                    B[k,l]=np.linalg.norm(error_matrices[:,:,k]@error_matrices[:,:,l])#,ord=inf)
            sol=np.zeros(5)
            sol[-1]=-1
            coefficients=np.linalg.solve(B,sol)
            P=0
            #print(B)
            #print(sol)
            for k in range(4):
                P+=coefficients[k]*Ps[:,:,k+1]
        """
        elif(i>1 and i<5):
            error_matrices=np.diff(Ps)
            B=np.zeros((i+1,i+1))
            for k in range(i+1):
                B[:,i]=-1
                B[i,:]=-1
            B[i,i]=0
            print(B.shape)
            for k in range(i):
                for l in range(i):
                    B[k,l]=np.max(abs(error_matrices[:,:,k]@error_matrices[:,:,l]))#,ord=inf)
            sol=np.zeros(i+1)
            sol[-1]=-1
            print(B)
            coefficients=np.linalg.solve(B,sol)
            P=0
            #print(B)
            #print(sol)
            for k in range(i):
                P+=coefficients[k]*Ps[:,:,k+1]
        """
        F=costruct_Fock_matrix(P,number_basissets,number_electrons,system,anti_symmetrize)
        epsilon, C = scipy.linalg.eigh(F)

        #print(C[0:4,0:4])
    return F,epsilon, C, converged

def compute_energy(C,F,system,number_electrons,number_basissets):
    energy=0
    P=construct_Density_matrix(C,number_electrons,number_basissets)
    for mu in range (2*number_basissets):
        for nu in range(2*number_basissets):
            energy+=P[nu,mu]*(system.h[nu,mu]+F[nu,mu])
    return energy*0.5

np.set_printoptions(precision=4)
l=10
grids_length=10
num_grid_points=1001
omega=0.25
a=0.25
odho = ODQD(l, grids_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))
number_electrons=2
anti_symmetrize=True
system=GeneralOrbitalSystem(n=number_electrons,basis_set=odho,anti_symmetrize=anti_symmetrize)
print("Reference energy: ",(system.compute_reference_energy()))
C=np.eye(2*l) #Create an initial guess for the coefficients
F,epsilon,C,converged=solve_DIIS(system,number_electrons,l,anti_symmetrize=anti_symmetrize,tol=1e-5,maxiter=30,C=C)
groundstate=C[:,0] #First column
print("Converged: ",converged)
print("Energy: ",compute_energy(C,F,system,number_electrons,l))
