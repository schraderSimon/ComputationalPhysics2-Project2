import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from quantum_systems import ODQD, GeneralOrbitalSystem
import scipy
import seaborn as sns
def construct_Density_matrix(C,number_electrons,l):
    n=np.eye(2*l)
    for i in range(number_electrons,2*l):
        n[i,i]=0
    P=C@n@C.conjugate().T
    """
    slicy=slice(0,number_electrons)
    P=np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy])
    """
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
                    #B[k,l]=np.max(abs(np.dot(error_matrices[:,:,k].flatten(),error_matrices[:,:,l].flatten())))#,ord=inf)
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
    tot_vec=system.h+F
    energy=0.5*np.einsum("nm,mn->",P,tot_vec)
    return energy
    """
    for mu in range (2*number_basissets):
        for nu in range(2*number_basissets):
            energy+=P[nu,mu]*(system.h[mu,nu]+F[mu,nu])
    return energy*0.5
    """
def find_psquared(system,C,number_basissets,num_grid_points):
    spin_up=system.spf[::2]
    spin_down=system.spf[1::2]
    C_up=C[::2,:]
    C_down=C[1::2,:]
    wf_up=np.zeros((2*l,num_grid_points),dtype=np.complex128)
    wf_down=np.zeros((2*l,num_grid_points),dtype=np.complex128)
    for i in range(2*number_basissets):
        for k in range(number_basissets):
            wf_up[i]+=(C_up[k,i]*spin_up[k])
            wf_down[i]+=(C_down[k,i]*spin_down[k])
    densities=(np.abs(wf_up)**2+np.abs(wf_down)**2)
    #densities=(np.abs(wf_up+wf_down)**2)
    return densities
def find_density(system,C,number_basissets,number_electrons,num_grid_points):
    P=construct_Density_matrix(C,number_electrons,number_basissets)
    density=np.zeros(len(system.spf[0]),dtype=np.complex128)
    for k in range(2*number_basissets):
        for l in range(2*number_basissets):
            density+=P[k,l]*np.conjugate(system.spf[k])*system.spf[l]
    return density
def get_dipole_matrix(system,number_basissets,num_grid_points,grid_length):
    """Create a 2*l matrix with M[mu,nu]=<mu|x|nu>"""
    """This is the AO basis"""
    l=number_basissets
    M=np.zeros((2*l,2*l),dtype=np.complex64)
    xvals=np.linspace(-grid_length,grid_length,num_grid_points)
    for i in range(2*l):
        for k in range(2*l):
            if i%2 != k%2: #If i and k are not both even or both odd. This is because the integral is necessarily zero when the two orbitals don't have the same spin.
                M[i,k]=0
                continue
            integrand=np.conj(system.spf[i])*xvals*(system.spf[k])
            M[i,k]=integrate_trapezoidal(integrand,2*grid_length/num_grid_points)
    print(M)
    return M
def get_dipole(C,system,number_basissets,num_grid_points,grid_length):
    P=construct_Density_matrix(C,number_electrons,number_basissets)
    dipole_matrix=get_dipole_matrix(system,number_basissets,num_grid_points,grid_length)
    return -np.einsum("mn,nm->",P,dipole_matrix)


@jit(nopython=True)
def integrate_trapezoidal(function_array,step):
    integral=0
    for i in range(1,len(function_array)-1):
        integral+=function_array[i]
    integral*=2
    integral+=function_array[0]+function_array[-1]
    integral*=step/2
    return integral
np.set_printoptions(precision=4)
l=10
grids_length=10
num_grid_points=1001
omega=0.25
a=0.25
steplength=(grids_length*2)/(num_grid_points-1)
odho = ODQD(l, grids_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))
number_electrons=2
anti_symmetrize=True
system=GeneralOrbitalSystem(n=number_electrons,basis_set=odho,anti_symmetrize=anti_symmetrize)
#print(system.spf)
print("Reference energy: ",(system.compute_reference_energy()))
C=np.eye(2*l) #Create an initial guess for the coefficients
F,epsilon,C,converged=solve(system,number_electrons,l,anti_symmetrize=anti_symmetrize,tol=1e-16,maxiter=500,C=C)
groundstate=C[:,0] #First column
print(groundstate)
print("Converged: ",converged)
print("Energy: ",compute_energy(C,F,system,number_electrons,l))

"""Test difference between P and F"""
P=construct_Density_matrix(C,number_electrons,l)
F=costruct_Fock_matrix(P,l,number_electrons,system,anti_symmetrize)
print("Absolute deviation: ",np.max(np.abs(P@F-F@P)))
print("Dipole: ",get_dipole(C,system,l,num_grid_points,grids_length))

"""Plot new basis densities and energies"""
newbasis=np.zeros((2*l,num_grid_points),dtype=np.complex128)
for i in range(2*l):
    for k in range(2*l):
        newbasis[i]+=(C[i,k]*system.spf[k])
fig = plt.figure(figsize=(16, 10))
potential = odho.HOPotential(omega=omega)
densities=find_psquared(system,C,l,num_grid_points)
plt.plot(system.grid, potential(system.grid))
for i in range(0,2*l,2):
    plt.plot(
        system.grid,
        np.abs(densities[i]) + epsilon[i].real,
        label=r"$\chi_{" + f"{i}" + r"}$",
    )

plt.grid()
plt.xlabel("Position [a.u.]")
plt.ylabel(r"Molecular orbit density + $\epsilon$")
plt.savefig("../figures/MO_densities.pdf")
plt.legend()
plt.show()

"""Plot electron density of the occupied system"""
ground_state_density=(densities[0]+densities[1])
print("Own ",integrate_trapezoidal(ground_state_density,step=2)*grids_length/(num_grid_points-1))

#density=find_density(system,C,l,number_electrons,num_grid_points)
#print("Øyvind ",integrate_trapezoidal(density,step=2)*grids_length/(num_grid_points-1))

fig, ax = plt.subplots(figsize=(16, 10))
ax.plot(system.grid,ground_state_density,label="HF ground state density")
#ax.plot(system.grid,density,label="Testerino")

ax.plot(system.grid,(system.spf[0]*np.conj(system.spf[0])+system.spf[1]*np.conj(system.spf[1])).real,label="Ground state with no interactions")
ax.set_xlim(-6,6)
ax.set_ylim(0,0.6)
im=plt.imread("ref.png")
ax.imshow(im,extent=[-6,6,0,0.4],aspect='auto')
plt.xlabel("distance [a.u.]")
plt.ylabel(r"electron density $\rho$")
plt.legend()
plt.xlabel("Position [a.u.]")
plt.ylabel(r"Total electron density $\rho$")
plt.savefig("../figures/total_density.pdf")
plt.show()
