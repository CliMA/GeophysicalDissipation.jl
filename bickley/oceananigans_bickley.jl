ENV["GKSwstype"] = "nul"
using Plots

using Printf
using Statistics
using CUDA

using Oceananigans
using Oceananigans.Advection
using Oceananigans.AbstractOperations
using Oceananigans.OutputWriters
using Oceananigans.Grids
using Oceananigans.Fields
using Oceananigans.Utils: prettytime

using GeophysicalDissipation.Bickley
using GeophysicalDissipation.DiskTimeSerieses: DiskTimeSeries

oceananigans_prefix(Nh, advection) = "oceananigans_bickley_Nh$(Nh)_$(typeof(advection).name.wrapper)"

function run(; Nh = 128,
               output_time_interval = 2,
               stop_time = 200,
               arch = CPU(),
               ν = 0,
               advection = WENO5())

    experiment_name = oceananigans_prefix(Nh, advection)

    grid = RegularCartesianGrid(size=(Nh, Nh, 1),
                                x = (-2π, 2π), y=(-2π, 2π), z=(0, 1),
                                topology = (Periodic, Periodic, Bounded))

    model = IncompressibleModel(architecture = arch,
                                 timestepper = :RungeKutta3, 
                                   advection = advection,
                                        grid = grid,
                                     tracers = :c,
                                     closure = IsotropicDiffusivity(ν=ν, κ=ν),
                                    buoyancy = nothing)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid
    
    # Parameters
    ϵ = 0.1 # perturbation magnitude
    ℓ = 0.5 # Gaussian width
    k = 0.5 # Sinusoidal wavenumber

    # Total initial conditions
    uᵢ(x, y, z) = Bickley.U(y) + ϵ * Bickley.ũ(x, y, ℓ, k)
    vᵢ(x, y, z) = ϵ * Bickley.ṽ(x, y, ℓ, k)
    cᵢ(x, y, z) = Bickley.C(y, grid.Ly)

    set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

    progress(sim) = @info(@sprintf("Iter: %d, time: %.1f, Δt: %.3f, max|u|: %.2f",
                                   sim.model.clock.iteration, sim.model.clock.time,
                                   sim.Δt.Δt, maximum(abs, u.data.parent)))

    wizard = TimeStepWizard(cfl=0.1, Δt=1e-4, max_change=1.1, max_Δt=10.0)

    simulation = Simulation(model, Δt=wizard, stop_time=stop_time,
                            iteration_interval=10, progress=progress)

    # Output: primitive fields + computations
    u, v, w, c = merge(model.velocities, model.tracers)

    ζ = ComputedField(∂x(v) - ∂y(u))

    outputs = merge(model.velocities, model.tracers, (ζ=ζ,))

    function init_grid_and_fields!(file, model)
        file["serialized/grid"] = model.grid
        
        for (i, field) in enumerate(outputs)
            field_name = keys(outputs)[i]
            file["timeseries/$field_name/meta/location"] = location(field)
        end

        return nothing
    end

    simulation.output_writers[:fields] =
        JLD2OutputWriter(model, outputs,
                                schedule = TimeInterval(output_time_interval),
                                init = init_grid_and_fields!,
                                prefix = experiment_name,
                                field_slicer = nothing,
                                force = true)

    @info "Running a simulation of an unstable Bickley jet with $(Nh)² degrees of freedom..."

    start_time = time_ns()

    run!(simulation)

    run_time = (time_ns() - start_time) * 1e-9

    @info "The simulation with Nh = $Nh and advection = $advection ran for $(prettytime(run_time))"
    @show cost = run_time / (model.clock.iteration * Nh^2)

    return experiment_name
end

function visualize(experiment_name)
    @info "Making a fun movie about an unstable Bickley jet..."

    filepath = experiment_name * ".jld2"

    ζ_timeseries = DiskTimeSeries(:ζ, filepath)
    c_timeseries = DiskTimeSeries(:c, filepath)

    grid = c_timeseries.grid

    xζ, yζ, zζ = nodes(ζ_timeseries)
    xc, yc, zc = nodes(c_timeseries)

    anim = @animate for (i, iteration) in enumerate(c_timeseries.iterations)

        @info "    Plotting frame $i from iteration $iteration..."
        
        ζ = ζ_timeseries[i]
        c = c_timeseries[i]
        t = ζ_timeseries.times[i]

        ζi = interior(ζ)[:, :, 1]
        ci = interior(c)[:, :, 1]

        kwargs = Dict(
                      :aspectratio => 1,
                      :linewidth => 0,
                      :colorbar => :none,
                      :ticks => nothing,
                      :clims => (-1, 1),
                      :xlims => (-grid.Lx/2, grid.Lx/2),
                      :ylims => (-grid.Ly/2, grid.Ly/2)
                     )

        ζ_plot = heatmap(xζ, yζ, clamp.(ζi, -1, 1)'; color = :balance, kwargs...)
        c_plot = heatmap(xc, yc, clamp.(ci, -1, 1)'; color = :thermal, kwargs...)

        ζ_title = @sprintf("ζ at t = %.1f", t)
        c_title = @sprintf("c at t = %.1f", t)

        plot(ζ_plot, c_plot, title = [ζ_title c_title], size = (4000, 2000))
    end

    gif(anim, experiment_name * ".gif", fps = 8)

    return nothing
end

experiment_name = run(Nh=32, arch=CPU(), advection=WENO5(), stop_time=200)
visualize(experiment_name)
