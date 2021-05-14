import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from quantum_systems import ODQD, GeneralOrbitalSystem
from helper_functions import integrate_trapezoidal
import scipy
import sys
import seaborn as sns
from numpy import sin, pi
from GHFSolver import *
np.set_printoptions(precision=1,linewidth=250)
def wrapperFunction(t,T=20*pi):
    returnval=sin(pi*t/T)**2*(t<T)
    return returnval
def timedependentPotential(t,somega=2,E=1):
    return E*sin(somega*t)
def total_Potential(t):
    val=wrapperFunction(t)*timedependentPotential(t)
    print(val)
    return wrapperFunction(t)*timedependentPotential(t)
number_electrons=2;
grid_length=10
num_grid_points=1001
omega=0.25
a=0.25
l=10
def plot_Comparison_HF(amount):
    energy_RHF=np.zeros(amount)
    energy_GHF=np.zeros(amount)
    solver=RHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(l));
    for i in range(amount):
        solver.solve(1e-14,1)
        energy_RHF[i]=solver.get_energy()
        print(i,energy_RHF[i])
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    for i in range(amount):
        solver.solve(1e-14,1)
        energy_GHF[i]=solver.get_energy()
        print(i,energy_GHF[i])
    sns.set_style("darkgrid")
    sns.set_context("talk")

    plt.plot(energy_RHF,label="RHF")
    plt.plot(energy_GHF,label="GHF")
    plt.legend()
    plt.xlabel("Number of steps in SCF algorithm")
    plt.ylabel("Energy (Hartree)")
    plt.tight_layout()
    plt.savefig("../../figures/comparison_RHF_GHF.pdf")
    plt.show()
plot_Comparison_HF(100)
sys.exit(1)

def plot_groundstate_densities():
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    solver.solve(1e-14,25)
    densities=solver.getDensity()
    ground_state_density_RHF=(densities[0]+densities[1])
    solver.solve(1e-14,5000)
    densities=solver.getDensity()
    ground_state_density_GHF=(densities[0]+densities[1])
    grid=solver.system.grid
    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(grid,ground_state_density_RHF,label="RHF ground state density")
    ax.plot(grid,ground_state_density_GHF,label="GHF ground state density")
    ax.set_xlim(-6,6)
    ax.set_ylim(0,0.4)
    #im=plt.imread("ref.png")
    #ax.imshow(im,extent=[-6,6,0,0.4],aspect='auto')

    plt.xlabel("distance [a.u.]")
    plt.ylabel(r"electron density $2\rho(x)$")
    plt.legend()
    plt.xlabel("Position [a.u.]")
    plt.ylabel(r"Total electron density $\rho$")
    plt.tight_layout()
    plt.savefig("../../figures/total_density.pdf")
    plt.show()
plot_groundstate_densities()
sys.exit(1)

def plot_molecular_orbitals():
    colors=["blue","orange","green","red"]
    potential=ODQD.HOPotential(omega=omega)
    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True,figsize=(15, 10))
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    grid=solver.system.grid
    solver.setC(np.eye(2*l));
    solver.solve(1e-14,25)
    densities=solver.getDensity()
    eigenvalues=solver.epsilon
    for i in range(0,4,1):
        ax1.axhline(eigenvalues[i].real,linestyle="--",color=colors[i])
        ax1.plot(
            grid,
            densities[i] + eigenvalues[i].real,
            label=r"$\rho_{" + f"{i}" + r"}$",color=colors[i]
        )
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    solver.solve(1e-14,5000)
    densities=solver.getDensity()
    eigenvalues=solver.epsilon
    for i in range(0,4,1):
        ax2.axhline(eigenvalues[i].real,linestyle="--",color=colors[i])
        ax2.plot(
            grid,
            densities[i] + eigenvalues[i].real,
            label=r"$\rho_{" + f"{i}" + r"}$",color=colors[i]
        )
    ax1.set_ylim(0,2)
    ax1.set_xlim(-10,10)
    ax2.set_xlim(-10,10)
    ax1.plot(grid,potential(grid),color="violet",linestyle="dotted")
    ax2.plot(grid,potential(grid),color="violet",linestyle="dotted")
    ax1.set_ylabel("Energy [Hartree] + electron density")
    ax1.set_xlabel("Position [a.u.]")
    ax2.set_xlabel("Position [a.u.]")
    ax1.set_title("RHF solution")
    ax2.set_title("GHF solution")
    ax1.legend()
    ax2.legend()
    plt.savefig("../../figures/MO_densities.pdf")
    plt.show()
#plot_molecular_orbitals()

def plot_time_evolution(time_potenial):
    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_ylim(0,1)
    ax.set_xlim(0,16)
    im=plt.imread("zanghellini2.png")
    #ax.imshow(im,extent=[0,4,0,1],aspect='auto')
    t=np.linspace(0,16*pi,2000)
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    solver.solve(1e-14,25)
    grid=solver.system.grid
    overlap=np.zeros(len(t))
    dipole_moment=np.zeros(len(t),dtype=np.complex128)
    energies=np.zeros(len(t))
    dt=t[1]-t[0]
    ground_state_densities=np.zeros((num_grid_points,len(t)))
    solver.init_TDHF(time_potenial)
    overlap[0]=solver.calculate_overlap()
    dipole_moment[0]=solver.getDipolemoment()
    for i,tval in enumerate(t[:-1]):
        solver.integrate(dt) #Integrate a step dt
        densities=solver.getDensity()
        ground_state_densities[:,i]=densities[0]
        overlap[i+1]=solver.calculate_overlap()
        dipole_moment[i+1]=solver.getDipolemoment()
    ax.plot(2*t/(2*pi),overlap,label="RHF")
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    solver.solve(1e-14,5000)
    grid=solver.system.grid
    overlap=np.zeros(len(t))
    dipole_moment=np.zeros(len(t),dtype=np.complex128)
    energies=np.zeros(len(t))
    dt=t[1]-t[0]
    ground_state_densities=np.zeros((num_grid_points,len(t)))
    solver.init_TDHF(time_potenial)
    overlap[0]=solver.calculate_overlap()
    dipole_moment[0]=solver.getDipolemoment()
    for i,tval in enumerate(t[:-1]):
        solver.integrate(dt) #Integrate a step dt
        densities=solver.getDensity()
        ground_state_densities[:,i]=densities[0]
        overlap[i+1]=solver.calculate_overlap()
        dipole_moment[i+1]=solver.getDipolemoment()
    ax.plot(t/pi,overlap,label="GHF")

    plt.legend()
    ax.set_xlabel(r"t $\omega/(2\pi)$")
    ax.set_ylabel(r"Overlap $|\langle\Psi(t)|\Psi(0)\rangle|^2$")
    plt.tight_layout()
    plt.savefig("../../figures/time_overlap.pdf")
    plt.show()
plot_time_evolution(timedependentPotential)





sns.set_style("darkgrid")
sns.set_context("talk")
plt.plot(t,dipole_moment)
plt.show()
sp=np.split(np.fft.fft(dipole_moment[2000:]),2)[0]
#sp=np.split(np.fft.fft(np.sin(2*np.pi*t[1500:]+1)),2)[0]
freq=np.split(np.fft.fftfreq(len(dipole_moment[2000:]),d=dt),2)[0]*(2*pi)
plt.plot(freq,sp.real,freq,sp.imag)
plt.plot(freq,np.abs(sp))
#plt.xlim(-1,1)
plt.show()

def plot_time_evolution():
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(t/pi,overlap,linewidth=2)
    ax.set_ylim(0,1)
    ax.set_xlim(0,4)
    im=plt.imread("zanghellini2.png")
    ax.imshow(im,extent=[0,4,0,1],aspect='auto')
    print(densities.shape)
    plt.show()
plot_time_evolution()

def animate():
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    line, = ax.plot(grid, np.sin(grid))
    def animate(i):
        line.set_ydata(ground_state_densities[:,i])
        return line,
    def init():
        line.set_ydata(np.ma.array(grid, mask=True))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(0,len(t)), init_func=init,
                                  interval=20, blit=True)
    plt.show()
#animate()
