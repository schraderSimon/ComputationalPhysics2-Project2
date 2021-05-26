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
def wrapperFunction_creator(T=1*pi):
    #Returns a function which returns 1 for t<T, zero otherwise
    def func(t):
        returnval=(t<T)#*sin(pi*t/T)**2 #looks like I do not need the extra stuff.
        return returnval
    return func
def timedependentPotential_creator(somega=2,E=1):
    #Returns the time dependent laser with values for omega and E
    def func(t):
        return E*sin(somega*t)
    return func
def total_Potential_creator(wrapper,potential):
    def func(t):
        return wrapper(t)*potential(t)
    return func

def plot_Comparison_HF(amount):
    energy_RHF=np.zeros(amount)
    energy_GHF=np.zeros(amount)
    solver=RHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(l));
    for i in range(amount):
        solver.solve(1e-17,1)
        energy_RHF[i]=solver.get_energy()
        print(i,energy_RHF[i])
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    for i in range(amount):
        solver.solve(1e-17,1)
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

def plot_groundstate_densities():
    solver =RHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(l));
    solver.solve(1e-17,25)
    C=solver.get_full_C()
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(C);
    densities=solver.getDensity()
    ground_state_density_RHF=(densities[0]+densities[1])
    solver.solve(1e-17,5000)
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

def plot_molecular_orbitals():
    colors=["blue","orange","green","red"]
    potential=ODQD.HOPotential(omega=omega)
    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig, (ax1, ax2) = plt.subplots(1, 2,sharey=True,figsize=(15, 10))
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    grid=solver.system.grid
    solver =RHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(l));
    solver.solve(1e-17,25)
    C=solver.get_full_C()
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(C);
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
    solver.solve(1e-17,5000)
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

def plot_time_evolution(time_potenial,animater=False,save_animation=False):
    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_ylim(0,1)
    ax.set_xlim(0,16)
    t=np.linspace(0,16*pi,2000)

    """RHF"""
    solver =RHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(l));
    solver.solve(1e-17,25)
    C=solver.get_full_C()
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(C);
    grid=solver.system.grid
    overlap_RHF=np.zeros(len(t))
    dipole_moment_RHF=np.zeros(len(t),dtype=np.complex128)
    energies=np.zeros(len(t))
    dt=t[1]-t[0]
    ground_state_densities_RHF=np.zeros((num_grid_points,len(t)))
    solver.init_TDHF(time_potenial)
    overlap_RHF[0]=solver.calculate_overlap()
    dipole_moment_RHF[0]=solver.getDipolemoment()
    for i,tval in enumerate(t[:-1]):
        solver.integrate(dt) #Integrate a step dt
        densities=solver.getDensity()
        ground_state_densities_RHF[:,i]=densities[0]+densities[1]
        overlap_RHF[i+1]=solver.calculate_overlap()
        dipole_moment_RHF[i+1]=solver.getDipolemoment()

    """GHF"""
    solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver.setC(np.eye(2*l));
    solver.solve(1e-17,5000)
    overlap_GHF=np.zeros(len(t))
    dipole_moment_GHF=np.zeros(len(t),dtype=np.complex128)
    energies=np.zeros(len(t))
    dt=t[1]-t[0]
    ground_state_densities_GHF=np.zeros((num_grid_points,len(t)))
    solver.init_TDHF(time_potenial)
    overlap_GHF[0]=solver.calculate_overlap()
    dipole_moment_GHF[0]=solver.getDipolemoment()
    for i,tval in enumerate(t[:-1]):
        solver.integrate(dt) #Integrate a step dt
        densities=solver.getDensity()
        ground_state_densities_GHF[:,i]=densities[0]+densities[1]
        overlap_GHF[i+1]=solver.calculate_overlap()
        dipole_moment_GHF[i+1]=solver.getDipolemoment()
    ax.plot(2*t/(2*pi),overlap_RHF,label="RHF")
    ax.plot(t/pi,overlap_GHF,label="GHF")

    plt.legend()
    ax.set_xlabel(r"t $\lambda\omega/(2\pi)$")
    ax.set_ylabel(r"Overlap $|\langle\Psi(t)|\Psi(0)\rangle|^2$")
    plt.tight_layout()
    plt.savefig("../../figures/time_overlap.pdf")
    plt.show()
    sns.set_style("darkgrid")
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0,16)
    plt.plot(t/pi,dipole_moment_RHF,label="RHF",zorder=5)
    plt.plot(t/pi,dipole_moment_GHF,label="GHF",zorder=3)
    plt.legend()
    plt.xlabel(r"t $\lambda\omega/(2\pi)$")
    plt.ylabel(r"Dipole moment")
    plt.tight_layout()
    plt.savefig("../../figures/TDDipolemoment_laser.pdf")
    plt.show()
    def animation():
        import matplotlib.animation as animation
        fig, ax = plt.subplots()
        line0, = ax.plot(grid, np.sin(grid),label="RHF")
        line1, = ax.plot(grid, np.sin(grid),label="GHF")
        time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        def animate(i):
            line0.set_ydata(ground_state_densities_RHF[:,i])
            line1.set_ydata(ground_state_densities_GHF[:,i])
            time_text.set_text(r'time =%.1f $\lambda\omega/(2\pi)$' % (t[i]/(2*pi)))

            return line0,line1,time_text,
        def init():
            line0.set_ydata(np.ma.array(grid, mask=True))
            line1.set_ydata(np.ma.array(grid, mask=True))
            time_text.set_text("")

            return line0,line1,time_text,

        ani = animation.FuncAnimation(fig, animate, np.arange(0,len(t)), init_func=init,
                                      interval=5, blit=True)
        ax.set_xlabel("Position [a.u.]")
        ax.set_ylabel(r"Total electron density $\rho$")
        ax.set_xlim(-10,10)
        ax.set_ylim(0,0.5)
        plt.legend()
        plt.tight_layout()
        if(save_animation):
            ani.save('../../figures/animation.gif',fps=400)
        plt.show()
    if animater:
        animation()
def plot_Fourier(potential,T):
    number_elements=1000
    total_simulation_time=200*pi+T
    wrapperFunction=wrapperFunction_creator(T)
    total_Potential=total_Potential_creator(wrapperFunction,potential)
    throwaway=int((number_elements)*T/total_simulation_time)+1
    if throwaway%2==1:
        throwaway+=1;
    t=np.linspace(0,total_simulation_time,number_elements)
    dt=t[1]-t[0]
    overlap_RHF=np.zeros(len(t))
    dipole_moment_RHF=np.zeros(len(t),dtype=np.complex128)
    overlap_GHF=np.zeros(len(t))
    dipole_moment_GHF=np.zeros(len(t),dtype=np.complex128)
    autocorrelation_RHF=np.zeros(len(t),dtype=np.complex128)
    autocorrelation_GHF=np.zeros(len(t),dtype=np.complex128)
    solver_RHF =RHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver_RHF.setC(np.eye(l));
    solver_RHF.solve(1e-17,25)
    C=solver_RHF.get_full_C()
    solver_RHF =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
    solver_RHF.setC(C);
    grid=solver_RHF.system.grid
    solver_RHF.init_TDHF(total_Potential)
    overlap_RHF[0]=solver_RHF.calculate_overlap()
    dipole_moment_RHF[0]=solver_RHF.getDipolemoment()
    counter=0
    for i,tval in enumerate(t[:-1]):
        if tval+dt>=T and counter==0:
            pass
            step=T-tval
            solver_RHF.integrate(step)
            solver_RHF.update_C0()
            counter+=1
            solver_RHF.integrate(dt-step)
        else:
            solver_RHF.integrate(dt) #Integrate a step dt
        overlap_RHF[i+1]=solver_RHF.calculate_overlap()
        dipole_moment_RHF[i+1]=solver_RHF.getDipolemoment()
        autocorrelation_RHF[i+1]=solver_RHF.autocorrelation()
    solver_GHF =solver_RHF
    solver_GHF.setC(np.eye(2*l));
    solver_GHF.solve(1e-17,5000)
    solver_GHF.init_TDHF(total_Potential)
    overlap_GHF[0]=solver_GHF.calculate_overlap()
    dipole_moment_GHF[0]=solver_GHF.getDipolemoment()
    counter=0
    for i,tval in enumerate(t[:-1]):
        if tval+dt>=T and counter==0:
            step=T-tval
            solver_GHF.integrate(step)
            solver_GHF.update_C0()

            counter+=1
            solver_GHF.integrate(dt-step)
        else:
            solver_GHF.integrate(dt) #Integrate a step dt
        overlap_GHF[i+1]=solver_GHF.calculate_overlap()
        dipole_moment_GHF[i+1]=solver_GHF.getDipolemoment()
        autocorrelation_GHF[i+1]=solver_GHF.autocorrelation()
    sns.set_style("darkgrid")
    sns.set_context("talk")
    plt.plot(t/pi,dipole_moment_RHF,label="RHF",zorder=5)
    plt.plot(t/pi,dipole_moment_GHF,label="GHF",zorder=3)
    plt.legend()
    plt.xlabel(r"t $\lambda\omega/(2\pi)$")
    plt.ylabel(r"Dipole moment $-|\langle\Psi(t)|\hat x|\Psi(t)\rangle|^2$")
    plt.tight_layout()
    plt.savefig("../../figures/TDDipolemoment.pdf")
    plt.show()
    sp_RHF=np.split(np.fft.fft(dipole_moment_RHF[throwaway:]),2)[0]
    sp_GHF=np.split(np.fft.fft(dipole_moment_GHF[throwaway:]),2)[0]
    freq=np.split(np.fft.fftfreq(len(dipole_moment_RHF[throwaway:]),d=dt),2)[0]*(2*pi) #2pi to go from frequency to angular frequency
    plt.plot(freq,np.abs(sp_RHF)/np.max(np.abs(sp_GHF)),label="RHF",zorder=5)
    plt.plot(freq,np.abs(sp_GHF)/np.max(np.abs(sp_GHF)),label="GHF",zorder=3)
    plt.xlim(0,1.0)
    plt.xlabel(r"Angular frequency")
    plt.ylabel("Relative intensity")
    plt.title("Dipole moment")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../figures/Fourier_spectrum.pdf")
    plt.show()
    #sp_RHF=np.split(np.fft.fft(autocorrelation_RHF[throwaway:]),2)[0]
    #sp_GHF=np.split(np.fft.fft(autocorrelation_GHF[throwaway:]),2)[0]
    sp_RHF=np.split(np.fft.fft(overlap_RHF[throwaway:]),2)[0]
    sp_GHF=np.split(np.fft.fft(overlap_GHF[throwaway:]),2)[0]
    freq=np.split(np.fft.fftfreq(len(dipole_moment_RHF[throwaway:]),d=dt),2)[0]*(2*pi) #2pi to go from frequency to angular frequency
    plt.plot(freq,np.abs(sp_RHF)/np.max(np.abs(sp_GHF)),label="RHF",zorder=5)
    plt.plot(freq,np.abs(sp_GHF)/np.max(np.abs(sp_GHF)),label="GHF",zorder=3)
    plt.xlim(-0.05,3)
    plt.xlabel(r"Angular frequency")
    plt.ylabel("Relative intensity")
    plt.title("Overlap")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../../figures/overlap_Fourier.pdf")
    plt.show()
    print(freq[1]-freq[0])

number_electrons=2;
grid_length=10
num_grid_points=1001
omega=0.25 #Larger omega -> shorter period (interesting!)
a=0.25
l=10
#plot_Comparison_HF(100)
#plot_groundstate_densities()
#plot_molecular_orbitals()
#plot_time_evolution(timedependentPotential_creator(somega=2,E=1),animater=True,save_animation=True)
plot_Fourier(timedependentPotential_creator(somega=2,E=1),pi)
