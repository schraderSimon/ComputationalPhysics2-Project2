import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from quantum_systems import ODQD, GeneralOrbitalSystem
from helper_functions import integrate_trapezoidal
import scipy
import seaborn as sns
from numpy import sin, pi
from GHFSolver import *

def timedependentPotential(t,somega=2,E=1):
    return E*sin(somega*t)

number_electrons=2;
grid_length=10
num_grid_points=1001
omega=0.25
a=0.25
l=10
solver =GHFSolverSystem(number_electrons,l,grid_length,num_grid_points,omega,a);
solver.setC(np.eye(2*l));
solver.solve(1e-5,25)
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
t=np.linspace(0,4*np.pi,2000)
overlap=np.zeros(len(t))
energies=np.zeros(len(t))
dt=t[1]-t[0]
ground_state_densities=np.zeros((num_grid_points,len(t)))
solver.init_TDHF(timedependentPotential)
for i,tval in enumerate(t):
    solver.integrate(dt) #Integrate a step dt
    densities=solver.getDensity()
    ground_state_densities[:,i]=densities[0]
    overlap[i]=solver.calculate_overlap()
grid=solver.system.grid
def plot_time_evolution():
    fig, ax = plt.subplots(figsize=(16, 10))

    ax.plot(t/np.pi,overlap,linewidth=2)
    ax.set_ylim(0,1)
    ax.set_xlim(0,4)
    im=plt.imread("zanghellini2.png")
    ax.imshow(im,extent=[0,4,0,1],aspect='auto')
    print(densities.shape)
    plt.show()

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

    ani = animation.FuncAnimation(fig, animate, np.arange(0, 2000), init_func=init,
                                  interval=2, blit=True)
    plt.show()
animate()
