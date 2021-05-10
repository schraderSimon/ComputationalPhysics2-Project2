using PyCall
using LinearAlgebra
using PyPlot

@pyimport quantum_systems as qs # imports the quantum_systems package for calculation of the particle orbitals.


struct QuantumLaserTrap1D2 # is a struct for 1-dimensional quantum laser trap systems with 2 particles.
    k::Float64 # is the strength of the harmonic trap potergy.
    a::Float64 # is the shielding of the Coulomb interaction between the particles.
    L::Float64 # is the amplitude of the laser field acting on the particles.
    ω::Float64 # is the frequency of the laser field acting on the particles.
end
QuantumLaserTrap1D2() = QuantumLaserTrap1D2(0.25^2,0.25,1.0,2pi)

function system_parameters(trap::QuantumLaserTrap1D2;type="slashes")::String
    # returns a string of the quantum laser trap parameters.
    if type == "slashes"
        return string("k = ",round(trap.k;digits=4)," / a = ",round(trap.a;digits=4),
            " / L = ",round(trap.L;digits=4)," / ω = ",round(trap.ω;digits=4))
    elseif type == "breaks"
        return "\n"*join((string(raw"$k = ",round(trap.k;digits=4),raw"$"),string(raw"$a = ",round(trap.a;digits=4),raw"$"),
            string(raw"$L = ",round(trap.L;digits=4),raw"$"),string(raw"$\omega = ",round(trap.ω;digits=4),raw"$")),"\n")*"\n"
    else
        return ""
    end
end



function find_HF_energy(trap::QuantumLaserTrap1D2=QuantumLaserTrap1D2();
        lattice_length::Float64=20.0, lattice_points::Int64=2001, orbitals::Int64=20, threshold::Float64=0.00001, iterations::Int64=10^6,
        text_output="full",plot_output="none")
    # finds an approximation to the ground state energy and system orbitals of the given quantum laser trap
    # by setting up a discretised and truncated one-body trap with the given number of orbitals using the quantum_systems package,
    # and then performing Hartree-Fock iteration until the given convergence threshold.

    # ASSERTIONS:

    if trap.k < 0
        error("Invalid value of the parameter k. The trap strength must be positive.")
    end
    if trap.a < 0
        error("Invalid value of the parameter a. The interaction shielding must be positive.")
    end
    if trap.L < 0
        error("Invalid value of the parameter L. The laser field amplitude must be positive.")
    end
    if trap.ω < 0
        error("Invalid value of the parameter ω. The laser field frequency must be positive.")
    end


    if orbitals < 2 || orbitals%2 == 1
        error("Invalid number of orbitals. Provide an even number above zero.")
    end

    if text_output ∉ ("full","none")
        error("The text output choice '",text_output,"' is not known. Choose between 'full' and 'none'.")
    end

    if plot_output ∉ ("particle orbitals","particle density","none")
        error("The text output choice '",text_output,"' is not known. ",
            "Choose between 'particle orbitals', 'particle density' and 'none'.")
    end


    # INITIAL OUTPUT:

    if text_output == "full"
        println()
        println("Finding the Hartree-Fock approximate ground state energy of a 1-dimensional quantum laser trap with 2 electrons.")
        println()
        println("System parameters: "*system_parameters(trap))
        println("Algorithm specifics: ",orbitals," particle orbitals / ",
            "Convergence threshold ",threshold," / Max ",iterations," iterations")
        println()
    end


    # CONSTANTS:

    k::Float64 = trap.k # is the strength of the harmonic trap potergy.
    a::Float64 = trap.a # is the shielding of the Coulomb interaction between the particles.
    L::Float64 = trap.L # is the amplitude of the laser field acting on the particles.
    ω::Float64 = trap.ω # is the frequency of the laser field acting on the particles.

    l::Int64 = orbitals # is the number of particle orbitals to be included in the Hartree-Fock calculation.

    if text_output == "full"
        println("Calculating particle orbital quantities using the quantum_systems package ...")
    end
    _U = qs.ODQD.HOPotential(omega=√k)
    odqd = qs.ODQD(l÷2,lattice_length/2;num_grid_points=lattice_points,alpha=1.0,a=a,potential=_U)
    odqd = qs.GeneralOrbitalSystem(2,odqd)
    χ = odqd.spf # are the particle orbitals (in discretised position basis).
    h = odqd.h # is the one-body Hamiltonian matrix (in particle orbital basis).
    u = odqd.u # is the two-body (anti-symmetrised) Coulomb interaction matrix (in particle orbital basis).
    x = odqd.position[1,:,:] # is the one-body position matrix (in particle orbital basis).
    lattice = odqd.grid # is the discretised lattice on which the above quantities were calculated.
    if text_output == "full"
        println("Particle orbital quantities calculated and stored!")
        println()
    end


    # VARIABLES:

    C::Matrix{ComplexF64} = ones(l,l) # is the coefficient matrix for the system orbitals (in particle orbital basis).
    D::Matrix{ComplexF64} = zeros(l,l) # is the density matrix for the system orbitals.

    F::Matrix{ComplexF64} = zeros(l,l) # is the Fock matrix for the system orbitals (in particle orbital basis).
    f::Vector{ComplexF64} = zeros(l) # is the vector of Fock eigenvalues for the system orbitals.
    E::Diagonal{ComplexF64} = Diagonal(f) # is the Fock eigenmatrix for the system orbitals.

    G::Float64 = 0.0 # is the to be calculated approximate ground state energy of the system.

    P::Vector{Float64} = zeros(lattice_points) # is the particle density (in discretised position basis).


    # FUNCTIONS:

    function find_system_orbitals!()
        # finds an approximation to the coefficient matrix through Hartree-Fock iteration.

        function update_D!()
            # updates the density matrix based on current coefficient matrix.
            D = zeros(l,l)
            for j in 1:l , i in 1:l
                for k in 1:2
                    D[i,j] += C[i,k]*C[j,k]'
                end
            end
        end

        function update_E!()
            # updates the Fock eigenmatrix based on the current Fock eigenvalues.
            E = Diagonal(f)
        end

        function update_F!()
            # updates the Fock matrix based on the current density matrix.
            F = h
            for j in 1:l , i in 1:l
                F += D[i,j]*u[:,j,:,i]
            end
        end

        update_D!()
        update_F!()
        if text_output == "full"
            println("Finding the system orbitals using Hartree-Fock iteration ...")
        end
        for i in 1:iterations
            f , C = eigen(F)
            update_D!()
            update_E!()
            update_F!()
            if maximum(abs2.(F*C-C*E)) < threshold^2
                if text_output == "full"
                    println("System orbitals found after ",i," iterations!")
                    println()
                end
                return
            end
        end
        error("No Hartree-Fock convergence even after ",iterations," iterations.")
    end

    function calculate_G!()
        # calculates the approximate ground state energy of the system.
        G = 0.0
        for i in 1:l , j in 1:l
            G += D[j,i]*h[i,j]
            for m in 1:l , n in 1:l
                G += D[m,i]*D[n,j]*u[i,j,m,n]
            end
        end
    end

    function calculate_P!()
        # calculates the particle density of the system (in discretised position basis).
        P = zeros(lattice_points)
        for n in 1:lattice_points
            for i in 1:l , j in 1:l
                P[n] += χ[i,n]'*D[i,j]*χ[j,n]
            end
        end
    end

    function plot_particle_orbitals()
        # plots the particle orbitals (in position basis) at their corresponding energy level.
        figure(figsize=(8,6))
        title("Particle orbitals of a 1D quantum laser trap"*"\n")
        plot(lattice,_U(lattice);color="#fdce0b")
        for i in 1:2:l
            plot(lattice,abs2.(χ[i,:]).+real(h[i,i]); label=raw"$\chi_{"*string(i)*raw"}$ & $\chi_{"*string(i+1)*raw"}$")
        end
        grid()
        xlabel(raw"$x$ [$\frac{4\pi\varepsilon}{me^2}$]")
        subplots_adjust(right=0.75)
        legend(title=system_parameters(trap;type="breaks"),bbox_to_anchor=(1.05,1.0))
    end

    function plot_particle_density()
        # plots the particle density (in discretised position basis).
        figure(figsize=(8,6))
        title("Particle density of a 1D quantum laser trap"*"\n")
        plot(lattice,_U(lattice);color="#fdce0b",label=raw"$U$ $\left[m\left(\frac{e^2}{4\pi\varepsilon\hbar}\right)^2\right]$")
        plot(lattice,real(P);label=raw"$P$")
        grid()
        xlim((-6,6))
        xlabel(raw"$x$ $\left[\frac{4\pi\varepsilon}{me^2}\right]$")
        ylim((0.0,1.0))
        subplots_adjust(right=0.75)
        legend(title=system_parameters(trap;type="breaks"),bbox_to_anchor=(1.05,1.0))
    end


    # EXECUTIONS:

    if plot_output == "particle orbitals"
        plot_particle_orbitals()
    end
    find_system_orbitals!()
    calculate_G!()
    if plot_output == "particle density"
        calculate_P!()
        plot_particle_density()
    end

    return G
end

; # suppresses inclusion output.
