import numpy as np
from quantum_systems import ODQD, GeneralOrbitalSystem
import scipy
from helper_functions import *
class GHFSolverSystem(object):
    '''
    General solver class for general Hartree Fock with HO-basis (1D) on a grid

    Attributes
    ---------
    l : int
        number of basis functions (2 times the argument)
    grid_length: float
        length of grid to solve HO wave function numerically
    num_grid_points: int
        number of points on the grid
    steplength: float
        distance between two grid points
    omega: double
        Harmonic Oscillator omega
    a: double
        shielding constant
    number_electrons: int
        The number of electrons
    system: GeneralOrbitalsystem
        The General orbital system constructed with the given parmeters
    C: 2D array (complex)
        The coefficient matrix. Initiated by "setC"
    C0: 2D array (complex)
        When starting to time evolve, the solution at time t=0.
        Initiated by init_TDHF
    integrator: scipy.integrate.complex_ode
        The integrator to proceed the equations
        Initiated by init_TDHF
    func: function(t)
        The time-dependent energy term
        Initiated by init_TDHF


    Methods
    ---------
    setC(C=None)
        Sets the sytem's C-matrix. Dimension needs to be correct (l*l)
        If C is None, a random matrix is used.
    construct_Density_matrix()
        Constructs the density matrix from self.C'
    construct_Fock_matrix()
        Constructs the Fock matrix from self.C
    solve(tol=1e-8,maxiter=100)
        Performs SCF algorithm (updating self.C) until ||P_new-P||<tol, or
        maxiter is reached
    get_energy()
        returns the system's energy from self.C
    getDensity()
        returns the system's electron density function (as 1D array)
    init_TDHF(func)
        Initializes the Time dependent solver with term "func"
    timeDependentFockMatrix(t,C)
        Returns the time-dependent Fock matrix at time t with coefficient matrix C
    RHS(t,C)
        The product -i F@C
    setUpIntegrator()
        Create an integrator object and initiate it
    integrate(dt)
        Propagate the coefficient matrix by a time dt
    autocorrelation()
        Calculate the autocorrelation <psi(0)|psi(t)>
    calculate_overlap()
        calculate the overlap |autocorrelation(t)|^2
    get_dipole_matrix()
        Calculate matrix containing dipoles in the basis set
    get_dipole_moment()
        Calculate the dipole moment
    reset_time()
        Revert time evolution
    '''
    def __init__(self,number_electrons=2,number_basisfunctions=10,
                grid_length=10,num_grid_points=1001,omega=0.25,a=0.25):
        """Initialize the system"""
        self.l=2*number_basisfunctions #Hartree Fock basis is twice the number of basis functions
        self.grid_length=grid_length
        self.num_grid_points=num_grid_points
        self.steplength=(grid_length*2)/(num_grid_points-1) #Step length for Numerical integration or similar
        self.omega=omega
        self.a=a
        self.number_electrons=number_electrons
        self.odho = ODQD(number_basisfunctions, grid_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))
        anti_symmetrize=True
        self.system=GeneralOrbitalSystem(n=number_electrons,basis_set=self.odho,
                                        anti_symmetrize=anti_symmetrize)
    def setC(self,C=None):
        """Initiate self.C-matrix and self.epsilon"""
        if C is None: #If C is none
            self.C=np.array(np.random.rand(self.l,self.l),dtype=complex128) #Random matrix (not a legal coefficient matrix)
        else:
            self.C=C
        P=self.construct_Density_matrix(self.C)
        F=self.construct_Fock_matrix(P)
        self.epsilon, throwaway = scipy.linalg.eigh(F)

    def construct_Density_matrix(self,C):
        """Construct and return density matrix from self.C"""
        slicy=slice(0,self.number_electrons)
        return np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy])

    def construct_Fock_matrix(self,P):
        """Construct Fock matrix"""
        u=np.einsum("ts,msnt->mn",P,self.system.u)
        return self.system.h+u

    def solve(self,tol=1e-8,maxiter=100):
        """Solve the SCF equations

        Parameters
        ----------
        tol: double
            The "convergence tolerance", system has converged if
            np.max(np.abs(P-P_old))<tol
        maxiter: int
            The maximum number of SCF iterations

        Returns
        -----
        bool
            if the system has converged or not after the proecedure has finished
        """
        converged=False #If difference is less than "tol"
        P=self.construct_Density_matrix(self.C)
        for i in range(maxiter):
            F=self.construct_Fock_matrix(P) #Before the if-test to assure it matches the recent Density matrix
            if(i>0):
                convergence_difference=np.max(np.abs(P-P_old))
                if (convergence_difference<tol):
                    converged=True
                    break
            P_old=P
            self.epsilon, self.C = scipy.linalg.eigh(F)
            P=self.construct_Density_matrix(self.C)
        return converged

    def get_energy(self):
        """Returns the ground-state energy <H>"""
        energy=0
        P=self.construct_Density_matrix(self.C)
        tot_vec=self.system.h+self.construct_Fock_matrix(P)
        energy=0.5*np.einsum("nm,mn->",P,tot_vec)
        return energy

    def getDensity(self):
        """Returns the ground state electron probability density as 1D array"""
        spin_up=self.system.spf[::2] #The atomic orbitals with spin up
        spin_down=self.system.spf[1::2] #The atomic orbitals with spin down
        C_up=self.C[::2,:] #The coefficient matrix for spin up
        C_down=self.C[1::2,:] #The coefficient matrix for spin down
        wf_up=np.zeros((self.l,self.num_grid_points),dtype=np.complex128) #The part of the wave function with spin up
        wf_down=np.zeros((self.l,self.num_grid_points),dtype=np.complex128)#The part of the wave function with spin down
        for i in range(self.l):
            for k in range(int(self.l/2)):
                wf_up[i]+=(C_up[k,i]*spin_up[k])
                wf_down[i]+=(C_down[k,i]*spin_down[k])
        densities=(np.abs(wf_up)**2+np.abs(wf_down)**2) #density is spin up density + spin down density
        return densities

    def init_TDHF(self,func):
        """Initiate the TDHF solver"""
        self.func=func
        self.C0=self.C # set C at time to to C0
        self.setUpIntegrator()

    def timeDependentFockMatrix(self,t,C):
        """Calculate the time dependent fock matrix"""
        if len(C.shape)==1: #If C is an array, transform to matrix
            C=C.reshape((self.l,self.l))
        P=self.construct_Density_matrix(C)
        F=self.construct_Fock_matrix(P)
        return F+self.system.position[0]*self.func(t)

    def RHS(self,t,C):
        """right hand side of the ODE, so dC/dT=RHS"""
        if len(C.shape)==1: #If C is an array, transform to matrix
            C=C.reshape((self.l,self.l))
        F=self.timeDependentFockMatrix(t,C)
        return -1j*np.ravel(F@C)

    def setUpIntegrator(self):
        """Initiate the integrator"""
        self.integrator=scipy.integrate.complex_ode(self.RHS).set_integrator("vode")
        self.integrator.set_initial_value(np.ravel(self.C),0)

    def integrate(self,dt):
        """Evolve in time by a time step dt, update self.C"""
        sol=self.integrator.integrate(self.integrator.t+dt).reshape((self.l,self.l))
        self.C=sol

    def autocorrelation(self):
        """Calculate autocorrelation <psi(t)|psi>"""
        occupied_0=self.C0[:,0:2] #Two electrons
        occupied_1=self.C[:,0:2] #Two electrons
        return np.linalg.det(np.conj(occupied_1).T@occupied_0)

    def calculate_overlap(self):
        """Calculate time-dependent overlap"""
        return np.abs(self.autocorrelation())**2

    def get_dipole_matrix(self):
        """Get position matrix"""
        M=np.zeros((self.l,self.l),dtype=np.complex64)
        xvals=np.linspace(-self.grid_length,self.grid_length,self.num_grid_points)
        for i in range(self.l):
            for k in range(self.l):
                if (i%2 != k%2):
                    M[i,k]=0
                    continue
                integrand=np.conj(self.system.spf[i])*xvals*(self.system.spf[k])
                M[i,k]=integrate_trapezoidal(integrand,2*self.grid_length/(self.num_grid_points-1))
        return M

    def getDipolemoment(self):
        """Calculate the dipole moment"""
        P=self.construct_Density_matrix(self.C)
        M=self.get_dipole_matrix()
        return -np.einsum("mn,nm->",P,M) #formula for calculation

    def reset_time(self):
        """Reset system"""
        self.C=self.C0 #Reset C
        self.setUpIntegrator() #Create new integrator
        print("test")

class RHFSolverSystem(GHFSolverSystem):
    """A restricted HF solver

    This class is NOT capable of time development, and many functions from the
    GHFSolverSystem do not work. Only the SCF-algorithm works,
    that is, constructing the SCF-C-matrix, and energy-construction.
    A call to the function "get_full_C", which then needs to be given as C-matrix
    to a GHFSolverSystem, is required for further analysis and applications.
    """
    def __init__(self,number_electrons=2,number_basisfunctions=10,
                grid_length=10,num_grid_points=1001,omega=0.25,a=0.25):
        self.l=number_basisfunctions #Hartree Fock basis is twice the number of basis functions
        self.grid_length=grid_length
        self.num_grid_points=num_grid_points
        self.steplength=(grid_length*2)/(num_grid_points-1) #Step length for Numerical integration or similar
        self.omega=omega
        self.a=a
        self.number_electrons=number_electrons
        self.system=ODQD(number_basisfunctions, grid_length, num_grid_points, a=a, alpha=1, potential=ODQD.HOPotential(omega))

    def construct_Density_matrix(self,C):

        slicy=slice(0,int(self.number_electrons/2)) #Slightly different expression
        return 2*np.einsum("ma,va->mv",C[:,slicy],C.conjugate()[:,slicy]) #Eq. 3.145

    def construct_Fock_matrix(self,P):
        """Construct Fock matrix, explicitely considering direct and exchange term"""
        udirect=np.einsum("ts,msnt->mn",P,self.system.u)
        uexchange=np.einsum("ts,mstn->mn",P,self.system.u)
        return self.system.h+udirect-0.5*uexchange #Also from Szabo-Ostlund

    def get_full_C(self):
        """Return the full coefficient matrix in the spin-orbital basis"""
        C=np.zeros((2*self.l,2*self.l),dtype=np.complex128)
        for i in range(self.l):
            for j in range(self.l):
                C[2*i,2*j]=self.C[i,j] #Same as RHS solution
                C[2*i+1,2*j+1]=self.C[i,j]
        return C
