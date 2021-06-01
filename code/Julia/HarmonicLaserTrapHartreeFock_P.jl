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
            " / L = ",round(trap.L;digits=4)," / λ = ",round(trap.λ;digits=4)," / T = ",(isinf(trap.T) ? "∞" : round(trap.T;digits=4)))
    elseif type == "breaks"
        return "\n"*join((string(raw"$\omega = ",round(trap.ω;digits=4),raw"$"),string(raw"$a = ",round(trap.a;digits=4),raw"$"),
            string(raw"$L = ",round(trap.L;digits=4),raw"$"),string(raw"$\lambda = ",round(trap.λ;digits=4),raw"$"),
            string(raw"$T = ",(isinf(trap.T) ? "∞" : round(trap.T;digits=4)),raw"$")),"\n")*"\n"
    else
        return ""
    end
end



function find_HF_state(trap::HarmonicLaserTrap1D2=HarmonicLaserTrap1D2(); particles::Int64 = 2,
        orbitals::Int64 = 20, lattice_length::Float64=20.0, lattice_points::Int64=1001,
        algorithm::String="GHF", iterations::Int64=10^6, threshold::Float64=0.1^(algorithm == "GHF" ? 8 : 6),
        text_output::String="full",plot_output::String="none")
    # finds an approximation to the ground state of the given harmonic laser trap with the given number of fermionic particles
    # by setting up a discretised and truncated one-body trap with the given number of orbitals using the quantum_systems package,
    # and then performing Hartree-Fock iteration with the given algorithm until the given convergence threshold.

    # ASSERTIONS:

    if trap.ω ≤ 0
        error("Invalid value of the parameter ω. The trap strength must be positive.")
    end
    if trap.a ≤ 0
        error("Invalid value of the parameter a. The interaction shielding must be positive.")
    end

    if algorithm ∉ ("GHF","RHF")
        error("The algorithm choice '",algorithm,"' is not known. Choose between 'general' (GHF) and 'spin-restricted' (RHF).")
    end

    if particles < 2 || particles%2 == 1 || particles > orbitals
        error("Invalid number of particles. Provide an even integer above zero, but no larger than the number of atomic basis orbitals.")
    end
    if orbitals < 2 || orbitals%2 == 1
        error("Invalid number of atomic basis orbitals. Provide an even integer above zero.")
    end

    if lattice_length ≤ 0
        error("Invalid spatial lattice length. Provide a positive value.")
    end
    if lattice_points < 3
        error("Invalid number of spatial lattice points. Provide an integer above 2.")
    end

    if iterations ≤ 0
        error("Invalid maximal number of Hartree-Fock iterations. Provide a positive integer.")
    end
    if threshold ≤ 0
        error("Invalid Hartree-Fock convergence threshold. Provide a positive value.")
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
        println("Finding the Hartree-Fock approximate ground state of a 1-dimensional harmonic laser trap with ",particles," electrons.")
        println()
        println("System parameters: "*system_parameters(trap))
        println("Algorithm specifics: ",(algorithm == "GHF" ? "General Hartree-Fock (GHF)" : "Spin-restricted Hartree-Fock (RHF)"),
            " algorithm /")
        println(orbitals," atomic orbitals ",
        " / ",lattice_points," spatial lattice points from –",round(lattice_length/2;digits=1)," to ",round(lattice_length/2;digits=1),
        " / Convergence threshold 0.1^",round(Int,-log10(threshold))," / Max ",iterations," iterations")
        println()
    end


    # CONSTANTS:

    ω::Float64 = trap.ω # is the strength of the harmonic trap potergy.
    a::Float64 = trap.a # is the shielding of the Coulomb interaction between the particles.

    N::Int64 = particles # is the number of fermionic particles in the harmonic trap.
    M::Int64 = orbitals # is the number of orbitals to be included in the Hartree-Fock calculation.

    if text_output != "none"
        println("Calculating atomic orbital quantities using the quantum_systems package ...")
    end
    _U = qs.ODQD.HOPotential(omega=ω)
    odqd = qs.ODQD(Int(M/2),lattice_length/2;num_grid_points=lattice_points,alpha=1.0,a=a,potential=_U)
    if algorithm == "GHF" # adds spin to the atomic orbital basis and antisymmetrises the interaction matrix
        # if general Hartree-Fock algorithm is chosen.
        odqd = qs.GeneralOrbitalSystem(N,odqd)
    elseif algorithm == "RHF" # redefines the dimensions M and N if the spin-restricted Hartree-Fock algorithm is chosen
        N /= 2
        M /= 2
    end
    χ = odqd.spf # are the atomic orbitals (in discretised position basis).
    h = odqd.h # is the one-body Hamiltonian matrix (in atomic orbital basis).
    u = odqd.u # is the two-body (anti-symmetrised) Coulomb interaction matrix (in atomic orbital basis).
    x = odqd.position[1,:,:] # is the one-body position matrix (in atomic orbital basis).
    lattice = odqd.grid # is the discretised lattice on which the above quantities were calculated.
    if text_output != "none"
        println("Atomic orbital quantities calculated and stored!")
        println()
    end

    algorithm_colour::String = (algorithm == "GHF" ? "#4aa888" : "#aa4888") # sets the colour of spatial density plots for the given algorithm.


    # VARIABLES:

    C::Matrix{ComplexF64} = zeros(M,N) # is the coefficient matrix for the molecular orbitals (in atomic orbital basis).
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
            if algorithm == "GHF"
                for b in 1:M , a in 1:M
                    F += P[a,b]*u[:,b,:,a]
                end
            elseif algorithm == "RHF"
                for b in 1:M , a in 1:M
                    F += P[a,b]*(2*u[:,b,:,a]-u[:,b,a,:])
                end
            end
        end

        if text_output != "none"
            println("Finding the molecular orbitals using Hartree-Fock iteration ...")
        end
        C = I(M)[1:M,1:N] # sets a cropped identity matrix as the initial guess for the coefficients.
        for i in 1:iterations
            update_P!()
            update_F!()
            if maximum(abs2.(F*P-P*F)) < threshold^2 && maximum(abs2.(C'*C-I(N))) < threshold^2
                # checks whether convergence of the Roothaan-Hall equation has been reached,
                # and assures that the molecular orbitals are orthonormal.
                if text_output != "none"
                    println("Molecular orbitals found after ",i," iterations!")
                    println()
                end
                return
            end
            C = eigvecs(F)[1:M,1:N]
        end
        println("– Warning: The Hartree-Fock algorithm did not converge even after ",iterations," iterations! –")
        println()
    end

    function calculate_HF_energy!()
        # calculates the Hartree-Fock approximate ground state energy of the system.
        E = 0.
        for a in 1:M , b in 1:M
            if algorithm == "GHF"
                E += P[a,b]*h[b,a]
                tmpf = 0.
                for c in 1:M , d in 1:M
                    tmpf += P[c,d]*u[b,d,a,c]
                end
                E += 1/2*P[a,b]*tmpf
            elseif algorithm == "RHF"
                E += 2*P[a,b]*h[b,a]
                tmpf = 0.
                for c in 1:M , d in 1:M
                    tmpf += P[c,d]*(2*u[b,d,a,c]-u[b,d,c,a])
                end
                E += P[a,b]*tmpf
            end
        end
    end

    function calculate_spatial_density!()
        # calculates the (discretised) spatial particle density of the system.
        ρ = zeros(lattice_points)
        for n in 1:lattice_points
            if algorithm == "GHF"
                for a in 1:2:M , b in 1:2:M
                    ρ[n] += χ[a,n]'*P[a,b]*χ[b,n]
                    ρ[n] += χ[a+1,n]'*P[a+1,b+1]*χ[b+1,n]
                end
            elseif algorithm == "RHF"
                for a in 1:M , b in 1:M
                    ρ[n] += 2*χ[a,n]'*P[a,b]*χ[b,n]
                end
            end
        end
    end

    function plot_atomic_orbitals()
        # plots the atomic orbitals (in discretised position basis) at their corresponding energy level.
        figure(figsize=(8,6))
        title(algorithm*" atomic orbitals of a 1D harmonic laser trap with "*string(particles)*" electrons"*"\n")
        plot(lattice,_U(lattice);color="#fdce0b")
        if algorithm == "GHF"
            for a in 1:2:M
                plot(lattice,abs2.(χ[a,:]).+real(h[a,a]);label=raw"$\chi_{"*string(a)*raw"}$ & $\chi_{"*string(a+1)*raw"}$")
            end
        elseif algorithm == "RHF"
            for a in 1:M
                plot(lattice,abs2.(χ[a,:]).+real(h[a,a]);
                    label=raw"$\chi^\downarrow_{"*string(a)*raw"}$ & $\chi^\uparrow_{"*string(a)*raw"}$")
            end
        end
        grid()
        xlabel(raw"$x$ [$\frac{4πϵ}{me^2}$]")
        subplots_adjust(right=0.75)
        legend(title=system_parameters(trap;type="breaks"),bbox_to_anchor=(1.05,1.0))
    end

    function plot_spatial_density()
        # plots the (discretised) spatial particle density of the system at its corresponding Hartree-Fock energy.
        figure(figsize=(8,6))
        title(algorithm*" spatial density of a 1D harmonic laser trap with "*string(particles)*" electrons"*
            "\n("*system_parameters(trap)*")\n")
        plot(lattice,_U(lattice);color="#fdce0b",label=raw"$U$ $\left[m\left(\frac{e^2}{4πϵ\hbar}\right)^2\right]$")
        plot(lattice,ρ.+E;label=raw"$\rho$",color=algorithm_colour)
        grid()
        xlim((-10,10))
        xlabel(raw"$x$ $\left[\frac{4πϵ}{me^2}\right]$")
        ylim((0.0,E+2.0))
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

    return E,P,χ,h,u,x
end

function find_HF_evolution(trap::HarmonicLaserTrap1D2=HarmonicLaserTrap1D2(); particles::Int64 = 2,
        orbitals::Int64 = 20, lattice_length::Float64=20.0, lattice_points::Int64=1001,
        algorithm::String="GHF", iterations::Int64=10^6, threshold::Float64=0.1^(algorithm == "GHF" ? 8 : 6),
        Δt::Float64= 2pi/trap.ω, time_resolution::Int64=1000,
        text_output::String="full", plot_output::String="fidelity")
    # finds an approximation to the ground state time evolution of the given harmonic laser trap
    # by setting up a discretised and truncated one-body trap with the given number of orbitals using the quantum_systems package,
    # performing Hartree-Fock iteration to get an initial Hartree-Fock ground state and then evolving this ground state
    # over the given time span using the time-dependent Roothaan-Hall equation for the given algorithm.

    # ASSERTIONS:

    if trap.L ≤ 0
        error("Invalid value of the parameter L. The laser field amplitude must be positive.")
    end
    if trap.λ ≤ 0
        error("Invalid value of the parameter λ. The relative frequency of the laser field must be positive.")
    end
    if trap.T ≤ 0
        error("Invalid value of the parameter T. The laser turn-off time must be positive or infinity.")
    end

    if Δt ≤ 0
        error("Invalid time span. Provide a positive value.")
    end
    if time_resolution < 2
        error("Invalid time resolution. Provide an integer value above 1.")
    end

    if text_output ∉ ("full","some","none")
        error("The text output choice '",text_output,"' is not known. Choose between 'full', 'some' and 'none'.")
    end
    if plot_output ∉ ("energy","dipole moment","fidelity")
        error("The plot output choice '",plot_output,"' is not known. ",
            "Choose between 'energy', 'dipole moment' and 'fidelity'.")
    end


    # CONSTANTS:

    ω::Float64 = trap.ω # is the strength of the harmonic trap potergy.
    L::Float64 = trap.L # is the amplitude of the laser field acting on the particles.
    λ::Float64 = trap.λ # is the relative frequency of the laser field acting on the particles.
    T::Float64 = trap.T # is the point in time at which the laser is turned off.

    E0,P0,χ,h,u,x = find_HF_state(trap;particles=particles,
        orbitals=orbitals,lattice_length=lattice_length,lattice_points=lattice_points,
        algorithm=algorithm,iterations=iterations,threshold=threshold,text_output=text_output)
        # are the energy and coefficient matrix of the initial Hartree-Fock ground state,
        # the atomic orbitals in (discretised spatial) basis, as well as the one-body Hamiltonian matrix,
        # the two-body (anti-symmetrised) Coulomb interaction matrix and the one-body position matrix, all in atomic orbital basis.

    N::Int64 = particles
    M::Int64 = orbitals
    if algorithm == "RHF"
        N /= 2
        M /= 2
    end

    parameters = (ω,L,λ,T,h,u,x,M,algorithm) # are the parameters involved for the time evolution.

    ts::Vector{Float64} = range(0.,Δt;length=time_resolution) # is the time lattice on which to plot the evolution observables.
    algorithm_colour::String = (algorithm == "GHF" ? "#4aa888" : "#aa4888") # sets the colour of ground state fidelity plots for the given algorithm.


    # VARIABLES:

    Ps::Vector{Matrix{ComplexF64}} = [zeros(M,M) for n in 1:time_resolution]
        # is the to-be-calculated density matrix evolution of the system.

    Γs::Vector{ComplexF64} = zeros(time_resolution) # is the to-be-calculated ground state fidelity evolution of the system.
    Es::Vector{ComplexF64} = zeros(time_resolution) # is the to-be-calculated energy evolution of the system.
    Ds::Vector{ComplexF64} = zeros(time_resolution) # is the to-be-calculated dipole moment evolution of the system.

    tmpf::ComplexF64 = 0. # (is a temporary float.)


    # FUNCTIONS:

    function evolve_P!(∂tP,P,parameters,t)
        ω,L,λ,T,h,u,x,M,algorithm = parameters

        F = h
        if t < T
            F -= x*L*sin(λ*ω*t)
        end
        if algorithm == "GHF"
            for b in 1:M , a in 1:M
                F += P[a,b]*u[:,b,:,a]
            end
        elseif algorithm == "RHF"
            for b in 1:M , a in 1:M
                F += P[a,b]*(2*u[:,b,:,a]-u[:,b,a,:])
            end
        end

        ∂tP .= im*(P*F-F*P)
    end

    function calculate_n_plot_fidelity!()
        # calculates and plots the ground state fidelity evolution of the system.
        for n in 1:time_resolution
            Γs[n] = 2*√(det(Ps[1])*det(Ps[n]))
            tmpf = 0.
            for a in 1:M , b in 1:M
                tmpf += Ps[1][a,b]*Ps[n][b,a]
            end
            Γs[n] += tmpf
        end
        figure(figsize=(8,6))
        title(algorithm*" ground state fidelity of a 1D harmonic laser trap with "*string(particles)*" electrons"*
            "\n("*system_parameters(trap)*")\n")
        plot(λ*ω/2pi*ts,real.(Γs);color=algorithm_colour,label=raw"$\Gamma$")
        plot(λ*ω/2pi*ts,imag.(Γs);linestyle="dotted",color=algorithm_colour)
        xlabel(raw"$\frac{2\pi}{\lambda\omega}t \quad \left[\frac{\hbar^3}{m}\left(\frac{4πϵ}{e^2}\right)^2\right]$")
        ylabel(raw"$\Gamma = \left|\langle\Phi_0|\Phi\rangle\right|^2$")
        grid()
    end

    function calculate_n_plot_energy!()
        # calculates and plots the energy evolution of the system.
        for n in 1:time_resolution
            for a in 1:M , b in 1:M
                Es[n] = Ps[n][a,b]*h[b,a]
                tmpf = 0.0
                for c in 1:M , d in 1:M
                    tmpf += Ps[n][c,d]*u[b,d,a,c]
                end
                Es[n] += 1/2*Ps[n][a,b]*tmpf
            end
            if algorithm == "RHF"
                Es *= 2
            end
        end
        figure(figsize=(8,6))
        title(algorithm*" expected energy of a 1D harmonic laser trap with "*string(particles)*" electrons"*
            "\n("*system_parameters(trap)*")\n")
        plot(λ*ω/2pi*ts,real.(Es);color="#fdce0b",label=raw"$E$")
        plot(λ*ω/2pi*ts,imag.(Es);linestyle="dotted",color="#fdce0b")
        xlabel(raw"$\frac{2\pi}{\lambda\omega}t \quad \left[\frac{\hbar^3}{m}\left(\frac{4πϵ}{e^2}\right)^2\right]$")
        ylabel(raw"$E \quad \left[\frac{m}{\hbar^2}\left(\frac{e^2}{4πϵ}\right)^2\right]$")
        grid()
    end

    function calculate_n_plot_dipole_moment!()
        # calculates and plots the dipole moment evolution of the system.
        for n in 1:time_resolution
            for a in 1:M , b in 1:M
                Ds[n] -= Ps[n][a,b]*x[b,a]
            end
            if algorithm == "RHF"
                Ds *= 2
            end
        end
        figure(figsize=(8,6))
        title(algorithm*" expected dipole moment of a 1D harmonic laser trap with "*string(particles)*" electrons"*
            "\n("*system_parameters(trap)*")\n")
        plot(λ*ω/2pi*ts,real.(Ds);color="#ff750a",label=raw"$D$")
        plot(λ*ω/2pi*ts,imag.(Ds);linestyle="dotted",color="#ff750a")
        xlabel(raw"$\frac{2\pi}{\lambda\omega}t \quad \left[\frac{\hbar^3}{m}\left(\frac{4πϵ}{e^2}\right)^2\right]$")
        ylabel(raw"$D \quad \left[\frac{4\pi\epsilon\hbar^2}{me}\right]$")
        grid()
    end


    # EXECUTIONS:

    if text_output != "none"
        println()
        println("Evolving the Hartree-Fock approximate ground state from t = 0.00 to t = ",round(Δt;digits=2)," ...")
    end
    ∂tP_function = ODEProblem(evolve_P!,P0,(0.,Δt),parameters)
    P_evolution = solve(∂tP_function)
    Ps = [P_evolution(ts[n]) for n in 1:time_resolution]
    if text_output != "none"
        println("Evolution calculated and stored!")
        println()
    end
    println("Calculating and plotting ground state fidelity ...")
    calculate_n_plot_fidelity!()
    if plot_output != "none"
        println("Calculating and plotting ",plot_output," ...")
        if plot_output == "energy"
            calculate_n_plot_energy!()
        elseif plot_output == "dipole moment"
            calculate_n_plot_dipole_moment!()
        end
    end
    println("Done! You are welcome.")
    println()

    return Γs,Ps,χ,h,u,x
end

; # suppresses inclusion output.
