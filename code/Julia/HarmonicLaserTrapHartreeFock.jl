using PyCall
using LinearAlgebra
using DifferentialEquations
using PyPlot

@pyimport quantum_systems as qs # imports the quantum_systems package for calculation of the atomic orbitals.


struct HarmonicLaserTrap1D2 # is a struct for 1-dimensional harmonic laser trap systems with 2 particles.
    ω::Float64 # is the strength of the harmonic trap potergy.
    a::Float64 # is the shielding of the Coulomb interaction between the particles.
    L::Float64 # is the amplitude of the laser field acting on the particles.
    λ::Float64 # is the relative frequency of the laser field acting on the particles.
    T::Float64 # is the point in time at which the is turned off.
end
HarmonicLaserTrap1D2(ω,a,L,λ) = HarmonicLaserTrap1D2(ω,a,L,λ,2pi/(λ*ω))
HarmonicLaserTrap1D2() = HarmonicLaserTrap1D2(0.25,0.25,1.0,8.0)

function system_parameters(trap::HarmonicLaserTrap1D2;type="slashes")::String
    # returns a string of the harmonic laser trap parameters.
    if type == "slashes"
        return string("ω = ",round(trap.ω;digits=4)," / a = ",round(trap.a;digits=4),
            " / L = ",round(trap.L;digits=4)," / λ = ",round(trap.λ;digits=4)," / T = ",round(trap.T;digits=4))
    elseif type == "breaks"
        return "\n"*join((string(raw"$\omega = ",round(trap.ω;digits=4),raw"$"),string(raw"$a = ",round(trap.a;digits=4),raw"$"),
            string(raw"$L = ",round(trap.L;digits=4),raw"$"),string(raw"$\lambda = ",round(trap.λ;digits=4),raw"$"),
            string(raw"$T = ",round(trap.T;digits=4)),raw"$"),"\n")*"\n"
    else
        return ""
    end
end



function find_HF_state(trap::HarmonicLaserTrap1D2=HarmonicLaserTrap1D2();
        lattice_length::Float64=20.0, lattice_points::Int64=2001, orbitals::Int64=20, threshold::Float64=0.1^8, iterations::Int64=10^8,
        text_output::String="full",plot_output::String="none")
    # finds an approximation to the ground state of the given harmonic laser trap
    # by setting up a discretised and truncated one-body trap with the given number of orbitals using the quantum_systems package,
    # and then performing Hartree-Fock iteration until the given convergence threshold.

    # ASSERTIONS:

    if trap.ω < 0
        error("Invalid value of the parameter ω. The trap strength must be positive.")
    end
    if trap.a < 0
        error("Invalid value of the parameter a. The interaction shielding must be positive.")
    end

    if orbitals < 2 || orbitals%2 == 1
        error("Invalid number of orbitals. Provide an even number above zero.")
    end

    if text_output ∉ ("full","some","none")
        error("The text output choice '",text_output,"' is not known. Choose between 'full', 'some' and 'none'.")
    end
    if plot_output ∉ ("atomic orbitals","spatial density","none")
        error("The plot output choice '",plot_output,"' is not known. ",
            "Choose between 'atomic orbitals', 'spatial density' and 'none'.")
    end


    # INITIAL OUTPUT:

    if text_output == "full"
        println()
        println("Finding the Hartree-Fock approximate ground state of a 1-dimensional harmonic laser trap with 2 electrons.")
        println()
        println("System parameters: "*system_parameters(trap))
        println("Algorithm specifics: ",orbitals," atomic orbitals / ",
            "Convergence threshold 0.1^",round(Int,-log10(threshold))," / Max ",iterations," iterations")
        println()
    end


    # CONSTANTS:

    ω::Float64 = trap.ω # is the strength of the harmonic trap potergy.
    a::Float64 = trap.a # is the shielding of the Coulomb interaction between the particles.

    M::Int64 = orbitals # is the number of orbitals to be included in the Hartree-Fock calculation.

    if text_output != "none"
        println("Calculating atomic orbital quantities using the quantum_systems package ...")
    end
    _U = qs.ODQD.HOPotential(omega=ω)
    odqd = qs.ODQD(Int(M/2),lattice_length/2;num_grid_points=lattice_points,alpha=1.0,a=a,potential=_U)
    odqd = qs.GeneralOrbitalSystem(2,odqd)
    χ = odqd.spf # are the atomic orbitals (in discretised position basis).
    h = odqd.h # is the one-body Hamiltonian matrix (in atomic orbital basis).
    u = odqd.u # is the two-body (anti-symmetrised) Coulomb interaction matrix (in atomic orbital basis).
    x = odqd.position[1,:,:] # is the one-body position matrix (in atomic orbital basis).
    lattice = odqd.grid # is the discretised lattice on which the above quantities were calculated.
    if text_output != "none"
        println("Atomic orbital quantities calculated and stored!")
        println()
    end


    # VARIABLES:

    C::Matrix{ComplexF64} = zeros(M,2) # is the coefficient matrix for the molecular orbitals (in atomic orbital basis).
    P::Matrix{ComplexF64} = zeros(M,M) # is the density matrix for the molecular orbitals.
    F::Matrix{ComplexF64} = zeros(M,M) # is the Fock matrix for the molecular orbitals (in atomic orbital basis).

    E::Float64 = 0. # is the to-be-calculated approximate ground state energy of the system.

    ρ::Vector{Float64} = zeros(lattice_points) # is the (discretised) spatial spatial density.

    tmpf::Float64 = 0. # (is a temporary float.)


    # FUNCTIONS:

    function find_molecular_orbitals!()
        # finds an approximation to the coefficient matrix through Hartree-Fock iteration.

        function update_P!()
            # updates the density matrix based on current coefficient matrix.
            P = C*C'
        end

        function update_F!()
            # updates the Fock matrix based on the current density matrix.
            F = h
            for b in 1:M , a in 1:M
                F += P[a,b]*u[:,b,:,a]
            end
        end

        if text_output != "none"
            println("Finding the molecular orbitals using Hartree-Fock iteration ...")
        end
        C = I(M)[1:M,1:2] # sets a cropped identity matrix as the initial guess for the coefficients.
        for i in 1:iterations
            update_P!()
            update_F!()
            if maximum(abs2.(F*P-P*F)) < threshold^2 && maximum(abs2.(C'*C-I(2))) < threshold^2
                # checks whether convergence of the Roothaan-Hall equation has been reached,
                # and assures that the molecular orbitals are orthonormal.
                if text_output != "none"
                    println("Molecular orbitals found after ",i," iterations!")
                    println()
                end
                return
            end
            C = eigvecs(F)[1:M,1:2]
        end
        println("– Warning: The Hartree-Fock algorithm did not converge even after ",iterations," iterations! –")
        println()
    end

    function calculate_HF_energy!()
        # calculates the Hartree-Fock approximate ground state energy of the system.
        E = 0.
        for a in 1:M , b in 1:M
            E += P[a,b]*h[b,a]
            tmpf = 0.
            for c in 1:M , d in 1:M
                tmpf += P[c,d]*u[b,d,a,c]
            end
            E += 1/2*P[a,b]*tmpf
        end
    end

    function calculate_spatial_density!()
        # calculates the (discretised) spatial particle density of the system.
        ρ = zeros(lattice_points)
        for n in 1:lattice_points
            for a in 1:2:M , b in 1:2:M
                ρ[n] += χ[a,n]'*P[a,b]*χ[b,n]
                ρ[n] += χ[a+1,n]'*P[a+1,b+1]*χ[b+1,n]
            end
        end
    end

    function plot_atomic_orbitals()
        # plots the atomic orbitals (in discretised position basis) at their corresponding energy level.
        figure(figsize=(8,6))
        title("Atomic orbitals of a 1D harmonic laser trap with 2 electrons"*"\n")
        plot(lattice,_U(lattice);color="#fdce0b")
        for a in 1:2:M
            plot(lattice,abs2.(χ[a,:]).+real(h[a,a]); label=raw"$\chi_{"*string(a)*raw"}$ & $\chi_{"*string(a+1)*raw"}$")
        end
        grid()
        xlabel(raw"$x$ [$\frac{4πϵ}{me^2}$]")
        subplots_adjust(right=0.75)
        legend(title=system_parameters(trap;type="breaks"),bbox_to_anchor=(1.05,1.0))
    end

    function plot_spatial_density()
        # plots the (discretised) spatial particle density of the system.
        figure(figsize=(8,6))
        title("Spatial density of a 1D harmonic laser trap with 2 electrons"*"\n")
        plot(lattice,_U(lattice);color="#fdce0b",label=raw"$U$ $\left[m\left(\frac{e^2}{4πϵ\hbar}\right)^2\right]$")
        plot(lattice,ρ;label=raw"$\rho$")
        grid()
        xlim((-6,6))
        xlabel(raw"$x$ $\left[\frac{4πϵ}{me^2}\right]$")
        ylim((0.0,1.0))
        subplots_adjust(right=0.75)
        legend(title=system_parameters(trap;type="breaks"),bbox_to_anchor=(1.05,1.0))
    end


    # EXECUTIONS:

    if plot_output == "atomic orbitals"
        plot_atomic_orbitals()
    end
    find_molecular_orbitals!()
    calculate_HF_energy!()
    if plot_output == "spatial density"
        calculate_spatial_density!()
        plot_spatial_density()
    end


    # FINAL OUTPUT:

    if text_output == "full"
        println("HF energy: ",round(E;digits=6))
        println()
    end


    return E,C,χ,h,u,x
end

function find_HF_evolution(trap::HarmonicLaserTrap1D2=HarmonicLaserTrap1D2(); Δt::Float64=201*trap.T, resolution::Int64=1000,
        text_output="full", plot_output="none")
    # finds an approximation to the ground state time evolution of the given harmonic laser trap
    # by setting up a discretised and truncated one-body trap with the given number of orbitals using the quantum_systems package,
    # performing Hartree-Fock iteration to get an initial Hartree-Fock ground state and then evolving this ground state with
    # the time-dependent Hartree-Fock equation.

    # ASSERTIONS:

    if trap.L < 0
        error("Invalid value of the parameter L. The laser field amplitude must be positive.")
    end
    if trap.λ < 0
        error("Invalid value of the parameter λ. The relative frequency of the laser field must be positive.")
    end
    if trap.T < 0
        error("Invalid value of the parameter T. The laser turn-off time must be positive or infinity.")
    end

    if text_output ∉ ("full","some","none")
        error("The text output choice '",text_output,"' is not known. Choose between 'full', 'some' and 'none'.")
    end
    if plot_output ∉ ("energy","dipole moment","density sum","none")
        error("The plot output choice '",plot_output,"' is not known. ",
            "Choose between 'energy','dipole moment', 'density sum' and 'none'.")
    end


    # CONSTANTS:

    ω::Float64 = trap.ω # is the strength of the harmonic trap potergy.
    L::Float64 = trap.L # is the amplitude of the laser field acting on the particles.
    λ::Float64 = trap.λ # is the relative frequency of the laser field acting on the particles.
    T::Float64 = trap.T # is the point in time at which the laser is turned off.

    E0,C0,χ,h,u,x = find_HF_state(trap;text_output=text_output)
        # are the energy and coefficient matrix of the initial Hartree-Fock ground state,
        # the atomic orbitals in (discretised spatial) basis, as well as the one-body Hamiltonian matrix,
        # the two-body (anti-symmetrised) Coulomb interaction matrix and the one-body position matrix, all in atomic orbital basis.

    M,_ = size(χ) # is the number of orbitals to be included in the Hartree-Fock calculations.

    parameters = (ω,L,λ,T,h,u,x) # are the parameters involved for the time evolution.

    ts::Vector{Float64} = range(0.,Δt;length=resolution) # is the time lattice on which to plot the evolution observables.


    # VARIABLES:

    Cs::Vector{Matrix{ComplexF64}} = [zeros(M,2) for n in 1:resolution] # is the to-be-calculated
    Ps::Vector{Matrix{ComplexF64}} = [zeros(M,M) for n in 1:resolution] # is the to-be-calculated density matrix evolution of the system.
    Es::Vector{ComplexF64} = zeros(resolution) # is the to-be-calculated energy evolution of the system.

    tmpf::ComplexF64 = 0. # (is a temporary float.)


    # FUNCTIONS:

    function evolve_C!(∂tC,C,parameters,t)
        ω,L,λ,T,h,u,x = parameters

        P = C*C'
        F = h
        if t < T
            F -= x*L*sin(2pi*t)
        end
        for b in 1:M , a in 1:M
            F += P[a,b]*u[:,b,:,a]
        end

        ∂tC = im*F*C
    end

    function plot_energy!()
        # calculates and plots the energy evolution of the system.
        for n in 1:resolution
            for a in 1:M , b in 1:M
                Es[n] += Ps[n][a,b]*h[b,a]
                tmpf = 0.0
                for c in 1:M , d in 1:M
                    tmpf += Ps[n][c,d]*u[b,d,a,c]
                end
                Es[n] += 1/2*Ps[n][a,b]*tmpf
            end
        end
        figure(figsize=(8,6))
        title("Expected energy of a 1D harmonic laser trap with 2 electrons"*"\n")
        plot(λ*ω/2pi*ts,real.(Es);color="#fdce0b")
        plot(λ*ω/2pi*ts,imag.(Es);linestyle="dotted",color="#fdce0b")
        xlabel(raw"$\frac{2\pi}{\lambda\omega}t$ $\left[\frac{\hbar^3}{m}\left(\frac{4πϵ}{e^2}\right)^2\right]$")
        ylabel(raw"$\langle E \rangle$ $\left[\frac{m}{\hbar^2}\left(\frac{e^2}{4πϵ}\right)^2\right]$")
    end

    function plot_density_sum!()
        # calculates and plots the density sum of the system.
        figure(figsize=(8,6))
        title("Density sum of a 1D harmonic laser trap with 2 electrons"*"\n")
        plot(λ*ω/2pi*ts,[real(sum(Ps[n])) for n in 1:resolution];color="#abcdef")
        plot(λ*ω/2pi*ts,[imag(sum(Ps[n])) for n in 1:resolution];linestyle="dotted",color="#abcdef")
        xlabel(raw"$\frac{2\pi}{\lambda\omega}t$ $\left[\frac{\hbar^3}{m}\left(\frac{4πϵ}{e^2}\right)^2\right]$")
    end


    # EXECUTIONS:

    if text_output != "none"
        println()
        println("Evolving the Hartree-Fock approximate ground state from t = 0.00 to t = ",round(Δt;digits=2)," ...")
    end
    ∂tC_function = ODEProblem(evolve_C!,C0,(0.,Δt),parameters)
    C_evolution = solve(∂tC_function)
    Cs = [C_evolution(ts[n]) for n in 1:resolution]
    Ps = [Cs[n]*Cs[n]' for n in 1:resolution]
    if text_output != "none"
        println("Evolution calculated and stored!")
        println()
    end
    if plot_output != "none"
        println("Calculating and plotting ",plot_output," ...")
        if plot_output == "energy"
            plot_energy!()
        elseif plot_output == "density sum"
            plot_density_sum!()
        end
        println("Done! You are welcome.")
        println()
    end

    return Cs,χ,h,u,x
end

; # suppresses inclusion output.
