using PyCall
using LinearAlgebra
using PyPlot

@pyimport quantum_systems as qs # imports the quantum_systems package for calculation of the atomic orbitals.


struct HarmonicLaserTrap1D2 # is a struct for 1-dimensional harmonic laser trap systems with 2 particles.
    ω::Float64 # is the strength of the harmonic trap potergy.
    a::Float64 # is the shielding of the Coulomb interaction between the particles.
    L::Float64 # is the amplitude of the laser field acting on the particles.
    λ::Float64 # is the relative frequency of the laser field acting on the particles.
end
HarmonicLaserTrap1D2() = HarmonicLaserTrap1D2(0.25,0.25,1.0,8.0)

function system_parameters(trap::HarmonicLaserTrap1D2;type="slashes")::String
    # returns a string of the harmonic laser trap parameters.
    if type == "slashes"
        return string("ω = ",round(trap.ω;digits=4)," / a = ",round(trap.a;digits=4),
            " / L = ",round(trap.L;digits=4)," / λ = ",round(trap.λ;digits=4))
    elseif type == "breaks"
        return "\n"*join((string(raw"$\omega = ",round(trap.ω;digits=4),raw"$"),string(raw"$a = ",round(trap.a;digits=4),raw"$"),
            string(raw"$L = ",round(trap.L;digits=4),raw"$"),string(raw"$\lambda = ",round(trap.λ;digits=4),raw"$")),"\n")*"\n"
    else
        return ""
    end
end



function find_HF_energy(trap::HarmonicLaserTrap1D2=HarmonicLaserTrap1D2();
        lattice_length::Float64=20.0, lattice_points::Int64=2001, orbitals::Int64=20, threshold::Float64=0.1^6, iterations::Int64=10^6,
        text_output="full",plot_output="none")
    # finds an approximation to the ground state energy and system orbitals of the given harmonic laser trap
    # by setting up a discretised and truncated one-body trap with the given number of orbitals using the quantum_systems package,
    # and then performing Hartree-Fock iteration until the given convergence threshold.

    # ASSERTIONS:

    if trap.ω < 0
        error("Invalid value of the parameter ω. The trap strength must be positive.")
    end
    if trap.a < 0
        error("Invalid value of the parameter a. The interaction shielding must be positive.")
    end
    if trap.L < 0
        error("Invalid value of the parameter L. The laser field amplitude must be positive.")
    end
    if trap.λ < 0
        error("Invalid value of the parameter λ. The relative frequency of the laser field must be positive.")
    end


    if orbitals < 2 || orbitals%2 == 1
        error("Invalid number of orbitals. Provide an even number above zero.")
    end

    if text_output ∉ ("full","none")
        error("The text output choice '",text_output,"' is not known. Choose between 'full' and 'none'.")
    end

    if plot_output ∉ ("atomic orbitals","spatial density","none")
        error("The plot output choice '",plot_output,"' is not known. ",
            "Choose between 'atomic orbitals', 'spatial density' and 'none'.")
    end


    # INITIAL OUTPUT:

    if text_output != "none"
        println()
        println("Finding the Hartree-Fock approximate ground state energy of a 1-dimensional harmonic laser trap with 2 electrons.")
        println()
        println("System parameters: "*system_parameters(trap))
        println("Algorithm specifics: ",orbitals," orbitals / ",
            "Convergence threshold 0.1^",round(Int,-log10(threshold))," / Max ",iterations," iterations")
        println()
    end


    # CONSTANTS:

    ω::Float64 = trap.ω # is the strength of the harmonic trap potergy.
    a::Float64 = trap.a # is the shielding of the Coulomb interaction between the particles.
    L::Float64 = trap.L # is the amplitude of the laser field acting on the particles.
    λ::Float64 = trap.λ # is the relative frequency of the laser field acting on the particles.

    l::Int64 = orbitals # is the number of orbitals to be included in the Hartree-Fock calculation.

    if text_output == "full"
        println("Calculating atomic orbital quantities using the quantum_systems package ...")
    end
    _U = qs.ODQD.HOPotential(omega=ω)
    odqd = qs.ODQD(Int(l/2),lattice_length/2;num_grid_points=lattice_points,alpha=1.0,a=a,potential=_U)
    odqd = qs.GeneralOrbitalSystem(2,odqd)
    χ = odqd.spf # are the atomic orbitals (in discretised position basis).
    h = odqd.h # is the one-body Hamiltonian matrix (in atomic orbital basis).
    u = odqd.u # is the two-body (anti-symmetrised) Coulomb interaction matrix (in atomic orbital basis).
    x = odqd.position[1,:,:] # is the one-body position matrix (in atomic orbital basis).
    lattice = odqd.grid # is the discretised lattice on which the above quantities were calculated.
    if text_output == "full"
        println("Atomic orbital quantities calculated and stored!")
        println()
    end


    # VARIABLES:

    C::Matrix{ComplexF64} = I(l) # is the coefficient matrix for the system orbitals (in atomic orbital basis).
    P::Matrix{ComplexF64} = zeros(l,l) # is the density matrix for the system orbitals.

    F::Matrix{ComplexF64} = zeros(l,l) # is the Fock matrix for the system orbitals (in atomic orbital basis).
    ε::Vector{ComplexF64} = zeros(l) # is the vector of Fock eigenvalues for the system orbitals.
    E::Diagonal{ComplexF64} = Diagonal(ε) # is the Fock eigenmatrix for the system orbitals.

    G::Float64 = 0.0 # is the to be calculated approximate ground state energy of the system.

    ρ::Vector{Float64} = zeros(lattice_points) # is the (discretised) spatial spatial density.

    tmp::Float64 = 0.0 # (is a temporary float.)

    # FUNCTIONS:

    function find_system_orbitals!()
        # finds an approximation to the coefficient matrix through Hartree-Fock iteration.

        function update_P!()
            # updates the density matrix based on current coefficient matrix.
            P = zeros(l,l)
            for j in 1:l , i in 1:l
                for k in 1:2
                    P[i,j] += C[i,k]*C[j,k]'
                end
            end
        end

        function update_E!()
            # updates the Fock eigenmatrix based on the current Fock eigenvalues.
            E = Diagonal(ε)
        end

        function update_F!()
            # updates the Fock matrix based on the current density matrix.
            F = h
            for j in 1:l , i in 1:l
                F += P[i,j]*u[:,j,:,i]
            end
        end

        update_P!()
        update_F!()
        if text_output == "full"
            println("Finding the system orbitals using Hartree-Fock iteration ...")
        end
        for i in 1:iterations
            ε , C = eigen(F)
            update_P!()
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
        println("– Warning: The Hartree-Fock algorithm did not convergence even after ",iterations," iterations! –")
        println()
    end

    function calculate_HF_energy!()
        # calculates the Hartree-Fock approximate ground state energy of the system.
        G = 0.0
        for i in 1:l , j in 1:l
            G += P[j,i]*h[i,j]
            tmp = 0.0
            for m in 1:l , n in 1:l
                tmp += P[m,i]*P[n,j]*u[i,j,m,n]
            end
            G += tmp/2
        end
    end

    function calculate_spatial_density!()
        # calculates the (discretised) spatial particle density of the system.
        ρ = zeros(lattice_points)
        for n in 1:lattice_points
            for i in 1:2:l , j in 1:2:l
                ρ[n] += χ[i,n]'*P[i,j]*χ[j,n]
                ρ[n] += χ[i+1,n]'*P[i+1,j+1]*χ[j+1,n]
            end
        end
    end

    function plot_atomic_orbitals()
        # plots the atomic orbitals (in discretised position basis) at their corresponding energy level.
        figure(figsize=(8,6))
        title("Atomic orbitals of a 1D harmonic laser trap with 2 electrons"*"\n")
        plot(lattice,_U(lattice);color="#fdce0b")
        for i in 1:2:l
            plot(lattice,abs2.(χ[i,:]).+real(h[i,i]); label=raw"$\chi_{"*string(i)*raw"}$ & $\chi_{"*string(i+1)*raw"}$")
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
    find_system_orbitals!()
    calculate_HF_energy!()
    if plot_output == "spatial density"
        calculate_spatial_density!()
        plot_spatial_density()
    end


    # FINAL OUTPUT
    if text_output != "none"
        println("HF energy: ",round(G;digits=6))
        println()
    end


    return G
end

; # suppresses inclusion output.
