using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SimulatedFields
using SimulatedFields: Circle, centre, inner_circle, outer_circle

using CairoMakie
using Dates
using Distributions
using FileIO
using LaTeXStrings
using MAT
using Random
using StaticArrays

set_theme!(theme_latexfonts())

datestring() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")

function rand_myelin_radii(p::TissueParameters, n...)
    return rand(Gamma(p.R_shape, p.R_scale), n...)
end

function rand_nonoverlapping_annulii(p::TissueParameters{T}; n = 10, width = 2.0, nattempts = 1000) where {T}
    annulii = Vector{Annulus{2, T}}(undef, n)
    n <= 0 && return annulii
    for _ in 1:nattempts
        for i in 1:n
            centre = SA[rand(Uniform(-T(width) / 2, T(width) / 2)), rand(Uniform(-T(width) / 2, T(width) / 2))]
            radius = rand_myelin_radii(p)
            annulii[i] = Annulus(; centre, radius, g_ratio = p.g_ratio)
        end
        any(isoverlapping(annulii[i], annulii[j]) for i in 1:n for j in i+1:n) && continue
        density = mean(1:10_000) do i
            x = SA[rand(Uniform(-T(width) / 2, T(width) / 2)), rand(Uniform(-T(width) / 2, T(width) / 2))]
            return any(x ∈ outer_circle(annulus) for annulus in annulii)
        end
        density < 0.65 && continue
        return annulii
    end
    return error("Failed to generate non-overlapping annulii after $nattempts attempts")
end

function rand_ferritin_spheres(p::TissueParameters{T}; n = 10, width = 2.0, nattempts = 1000) where {T}
    spheres = Vector{Circle{3, T}}(undef, n)
    n <= 0 && return spheres
    for _ in 1:nattempts
        for i in 1:n
            centre = T(0.95) .* SA[rand(Uniform(-T(width) / 2, T(width) / 2)), rand(Uniform(-T(width) / 2, T(width) / 2)), zero(T)]
            radius = p.R_Ferritin
            spheres[i] = Circle(; centre, radius)
        end
        (n == 1 || minimum(norm(centre(spheres[i]) - centre(spheres[j])) for i in 1:n for j in i+1:n) > width / 4) && break
    end
    return spheres
end

function myelin_plot(; width = 1.2, ngrid = 4096, theta = 0.0, nmyelin = 5, nferritin = 3, randmyelin = nmyelin > 1, randferritin = nferritin > 1, nattempts = 100_000_000, units = "Hz", save = true, ext = ".png", kwargs...)
    @assert ispow2(ngrid) "ngrid must be a power of 2"
    x = range(-width / 2, width / 2; length = ngrid)
    y = range(-width / 2, width / 2; length = ngrid)
    p = TissueParameters{Float64}(; theta = deg2rad(theta), kwargs...)

    annulii = randmyelin ?
              rand_nonoverlapping_annulii(p; n = nmyelin, width = width, nattempts = nattempts) :
              nmyelin == 1 ? [Annulus(; centre = SA[0.0, 0.0], radius = p.R_mu, g_ratio = p.g_ratio)] :
              (@assert nmyelin == 0; Annulus{2, Float64}[])
    spheres = randferritin ?
              rand_ferritin_spheres(p; n = nferritin, width = width) :
              nferritin == 1 ? [Circle{3, Float64}(; centre = SA[0.0, 0.0, 0.0], radius = p.R_Ferritin)] :
              (@assert nferritin == 0; Circle{3, Float64}[])

    # Note: could use `omega(x, y, p, annulii, spheres)` directly, but this way we get the bounds for the colorbar,
    # otherwise the ferritin fields will dominate and you won't see the myelin fields
    ω_myelin = omega(x, y, p, annulii, Circle{3, Float64}[])
    ω_ferritin = omega(x, y, p, Annulus{2, Float64}[], spheres)
    ω_total = ω_myelin .+ ω_ferritin

    ω_bounds = nferritin > 0 ? 2π .* (-10.0, 10.0) : maximum(abs, ω_myelin) .* (-1, 1) # extrema(ω_myelin) # maximum(abs, ω_total) .* (-1, 1)
    ω_rescale = lowercase(units) == "hz" ? 2π : 1.0
    ω_clamped = clamp.(ω_total, ω_bounds...)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x [μm]", ylabel = "y [μm]", aspect = 1.0, title = L"Field Angle $\theta = %$(round(theta; digits = 1))\degree$")
    hm = heatmap!(ax, x, y, ω_clamped ./ ω_rescale; colormap = :jet, colorrange = ω_bounds ./ ω_rescale)
    Colorbar(fig[1, 2], hm; label = lowercase(units) == "hz" ? "Frequency [Hz]" : "ω [rad/s]")

    if save === true || save === :prompt
        if save === :prompt
            display(fig)
            println("Save plot and data? [y/N]")
            lowercase(readline()) != "y" && return fig
        end
        timestamp = datestring()
        savedir = mkpath(joinpath(@__DIR__, "output"))
        basename = nferritin == 0 ? "myelin-field" : nmyelin == 0 ? "ferritin-field" : "myelin-field-with-ferritin"
        theta_suff = string("theta-", theta == round(Int, theta) ? string(Int(theta)) : string(theta))

        # Save plot
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ext))
        FileIO.save(filename, fig)

        # Save data
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ".mat"))
        ndown = ngrid > 512 ? ngrid ÷ 512 : 1
        Idown = ndown:ndown:ngrid
        data = Dict{String, Any}()
        data["params"] = Dict(p)
        if nferritin == 0
            data["omega_myelin"] = ω_myelin[Idown, Idown]
        elseif nmyelin == 0
            data["omega_ferritin"] = ω_ferritin[Idown, Idown]
        else
            data["omega_myelin"] = ω_myelin[Idown, Idown]
            data["omega_ferritin"] = ω_ferritin[Idown, Idown]
            data["omega_total"] = ω_total[Idown, Idown]
        end
        matwrite(filename, data)
    end

    return fig
end

function vessel_plot(; width = 1.0, ngrid = 4096, theta = 0.0, radius = 0.1, units = "Hz", save = true, ext = ".png")
    @assert ispow2(ngrid) "ngrid must be a power of 2"
    x = range(-width / 2, width / 2; length = ngrid)
    y = range(-width / 2, width / 2; length = ngrid)
    p = TissueParameters{Float64}(; theta = deg2rad(theta))

    circles = [Circle{2, Float64}(; centre = SA[0.0, 0.0], radius = radius)]
    ω_vessels = omega(x, y, p, circles)

    ω_bounds = maximum(abs, ω_vessels) .* (-1, 1) # extrema(ω_vessels)
    ω_rescale = lowercase(units) == "hz" ? 2π : 1.0
    ω_clamped = clamp.(ω_vessels, ω_bounds...)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x [mm]", ylabel = "y [mm]", aspect = 1.0, title = L"Field Angle $\theta = %$(round(theta; digits = 1))\degree$")
    hm = heatmap!(ax, x, y, ω_clamped ./ ω_rescale; colormap = :jet, colorrange = ω_bounds ./ ω_rescale)
    Colorbar(fig[1, 2], hm; label = lowercase(units) == "hz" ? "Frequency [Hz]" : "ω [rad/s]")

    if save === true
        timestamp = datestring()
        savedir = mkpath(joinpath(@__DIR__, "output"))
        basename = "vessel-field"
        theta_suff = string("theta-", theta == round(Int, theta) ? string(Int(theta)) : string(theta))

        # Save plot
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ext))
        FileIO.save(filename, fig)

        # Save data
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ".mat"))
        ndown = ngrid > 512 ? ngrid ÷ 512 : 1
        Idown = ndown:ndown:ngrid
        data = Dict{String, Any}()
        data["params"] = Dict(p)
        data["omega_vessel"] = ω_vessels[Idown, Idown]
        matwrite(filename, data)
    end

    return fig
end

function make_plots(; seed = 30, save = true)
    # Vessel plot
    for theta in 0:10:90
        vessel_plot(; width = 1.0, theta, radius = 0.1, save = save) |> display
    end

    # Myelin plots
    for theta in 0:10:90
        Random.seed!(seed)
        myelin_plot(; width = 1.0, theta, nmyelin = 4, nferritin = 0, save = save) |> display
    end

    # Ferritin plot
    myelin_plot(; width = 0.5, theta = 90.0, nmyelin = 0, nferritin = 1, save = save) |> display

    return nothing
end
