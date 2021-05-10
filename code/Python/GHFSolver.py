import numpy as np
from quantum_systems import ODQD, GeneralOrbitalSystem
import scipy
class GHFSolverSystem(object):
    def __init__(self,number_electrons=2,number_basisfunctions=10,grid_length=10,num_grid_points=1001,omega=0.25,a=0.25):
        self.l=2*number_basisfunctions
        self.grid_length=grid_length
        self.num_grid_points=num_grid_points
        self.steplength=(grid_length*2)/(num_grid_points-1)
        self.omega=omega
        self.a=a
        self.number_electrons=number_electrons
        odho = ODQD(number_basisfunctions, grid_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))
        anti_symmetrize=True
        self.system=GeneralOrbitalSystem(n=number_electrons,basis_set=odho,anti_symmetrize=anti_symmetrize)
    def setC(self,C=None):
        if C is None:
            self.C=np.random.rand(self.l,self.l) #Random matrix (possibly illegal)
        self.C=C
    def construct_Density_matrix(self,C):
        slicy=slice(0,self.number_electrons)
        return np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy])
    def construct_Fock_matrix(self,P):
        u=np.einsum("ts,msnt->mn",P,self.system.u)
        return self.system.h+u
    def solve(self,tol=1e-8,maxiter=100):
        converged=False
        P=self.construct_Density_matrix(self.C)
        for i in range(maxiter):
            F=self.construct_Fock_matrix(P) #Before the if-test to assure it matches the recent Density matrix
            if(i>0):
                convergence_difference=np.max(np.abs(P-P_old))
                if (convergence_difference<tol):
                    converged=True
                    break
            P_old=P
            epsilon, self.C = scipy.linalg.eigh(F)
            P=self.construct_Density_matrix(self.C)
        return converged

    def get_energy(self):
        energy=0
        P=construct_Density_matrix(self.C)
        tot_vec=self.system.h+self.construct_Fock_matrix(P)
        energy=0.5*np.einsum("nm,mn->",P,tot_vec)
        print(energy)
        return energy
    def getDensity(self):
        spin_up=self.system.spf[::2]
        spin_down=self.system.spf[1::2]
        C_up=self.C[::2,:]
        C_down=self.C[1::2,:]
        wf_up=np.zeros((self.l,self.num_grid_points),dtype=np.complex128)
        wf_down=np.zeros((self.l,self.num_grid_points),dtype=np.complex128)
        for i in range(self.l):
            for k in range(int(self.l/2)):
                wf_up[i]+=(C_up[k,i]*spin_up[k])
                wf_down[i]+=(C_down[k,i]*spin_down[k])
        densities=(np.abs(wf_up)**2+np.abs(wf_down)**2)
        return densities
    def init_TDHF(self,func):
        self.func=func
        self.C0=self.C # set C at time to to C0
        self.setUpIntegrator()
    def timeDependentFockMatrix(self,t,C):
        if len(C.shape)==1:
            C=C.reshape((self.l,self.l))
        P=self.construct_Density_matrix(C)
        F=self.construct_Fock_matrix(P)
        return F+self.system.position[0]*self.func(t)
    def RHS(self,t,C):
        if len(C.shape)==1:
            C=C.reshape((self.l,self.l))
        F=self.timeDependentFockMatrix(t,C)
        return -1j*np.ravel(F@C)
    def setUpIntegrator(self):
        self.integrator=scipy.integrate.complex_ode(self.RHS).set_integrator("vode")
        self.integrator.set_initial_value(np.ravel(self.C),0)
    def integrate(self,dt):
        sol=self.integrator.integrate(self.integrator.t+dt).reshape((self.l,self.l))
        self.C=sol
    def calculate_overlap(self):
        occupied_0=self.C0[:,0:2]
        occupied_1=self.C[:,0:2]
        return np.abs(np.linalg.det(np.conj(occupied_1).T@occupied_0))**2
    def reset_time(self):
        self.C=self.C0
