import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from quantum_systems import ODQD, GeneralOrbitalSystem
from helper_functions import integrate_trapezoidal
import scipy
import seaborn as sns
from numpy import sin, pi
from GHFSolver import *
def wrapperFunction(t,T=10*pi):
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
omega=1
a=0.25
l=10
solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
solver.setC(np.eye(2*l));
solver.solve(1e-16,25)
def plot_groundstate_density(solver):
    densities=solver.getDensity()
    ground_state_density=(densities[0]+densities[1])
    grid=solver.system.grid
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(grid,ground_state_density,label="HF ground state density")
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
plot_groundstate_density(solver)
t=np.linspace(0,20*pi,5000)
overlap=np.zeros(len(t))
dipole_moment=np.zeros(len(t),dtype=np.complex128)
energies=np.zeros(len(t))
dt=t[1]-t[0]
ground_state_densities=np.zeros((num_grid_points,len(t)))
solver.init_TDHF(total_Potential)
for i,tval in enumerate(t):
    solver.integrate(dt) #Integrate a step dt
    densities=solver.getDensity()
    ground_state_densities[:,i]=densities[0]
    overlap[i]=solver.calculate_overlap()
    dipole_moment[i]=solver.getDipolemoment()
grid=solver.system.grid
plt.plot(t,dipole_moment)
plt.show()
sp=np.split(np.fft.fft(dipole_moment[2500:]),2)[0]
#sp=np.split(np.fft.fft(np.sin(2*np.pi*t[1500:]+1)),2)[0]
freq=np.split(np.fft.fftfreq(len(dipole_moment[2500:]),d=dt),2)[0]*(2*pi)
plt.plot(freq,sp.real,freq,sp.imag)
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
animate()
