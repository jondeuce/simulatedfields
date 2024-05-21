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

function myelin_plot(; width = 1.2, ngrid = 4096, theta = 0.0, nmyelin = 5, nferritin = 3, randmyelin = nmyelin > 1, randferritin = nferritin > 1, nattempts = 100_000_000, units = "rad/s", save = true, plot = true, ext = ".png", kwargs...)
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

    # Note: could use `omegamap(x, y, p, annulii, spheres)` directly, but this way we get the bounds for the colorbar,
    # otherwise the ferritin fields will dominate and you won't see the myelin fields
    myelindomain = MyelinDomain(annulii, Circle{3, Float64}[])
    ferritindomain = MyelinDomain(Annulus{2, Float64}[], spheres)

    ω_myelin = omegamap(x, y, p, myelindomain)
    ω_ferritin = omegamap(x, y, p, ferritindomain)
    ω_total = ω_myelin .+ ω_ferritin

    region_myelin = regionmap(x, y, p, myelindomain)
    t1_myelin = t1map(x, y, p, myelindomain)
    t2_myelin = t2map(x, y, p, myelindomain)
    region_ferritin = regionmap(x, y, p, ferritindomain)
    t1_ferritin = t1map(x, y, p, ferritindomain)
    t2_ferritin = t2map(x, y, p, ferritindomain)

    ω_bounds = nferritin > 0 ? 2π .* (-10.0, 10.0) : maximum(abs, ω_myelin) .* (-1, 1) # extrema(ω_myelin) # maximum(abs, ω_total) .* (-1, 1)
    ω_rescale = lowercase(units) == "hz" ? 2π : 1.0
    ω_clamped = clamp.(ω_total, ω_bounds...)
    ω_label = string("Frequency [", lowercase(units) == "hz" ? "Hz" : "rad/s", "]")

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x [μm]", ylabel = "y [μm]", aspect = 1.0, title = L"Field Angle $\theta = %$(round(theta; digits = 1))\degree$")
    hm = heatmap!(ax, x, y, ω_clamped ./ ω_rescale; colormap = :jet, colorrange = ω_bounds ./ ω_rescale)
    Colorbar(fig[1, 2], hm; label = ω_label)
    plot && display(fig)

    mosaic = Figure()
    ax1 = Axis(mosaic[1, 1]; aspect = 1.0, title = L"%$(ω_label): $\theta = %$(round(theta; digits = 1))\degree$", ylabel = "y [μm]")
    hm1 = heatmap!(ax1, x, y, ω_clamped ./ ω_rescale; colormap = :jet, colorrange = ω_bounds ./ ω_rescale)
    Colorbar(mosaic[1, 2], hm1)
    ax2 = Axis(mosaic[1, 3]; aspect = 1.0, title = L"\text{Region}")
    hm2 = heatmap!(ax2, x, y, nferritin == 0 ? region_myelin : region_ferritin; colormap = :jet, colorrange = (0, 4))
    Colorbar(mosaic[1, 4], hm2)
    ax3 = Axis(mosaic[2, 1]; aspect = 1.0, title = L"$T_1$ [s]", xlabel = "x [μm]", ylabel = "y [μm]")
    hm3 = heatmap!(ax3, x, y, nferritin == 0 ? t1_myelin : t1_ferritin; colormap = :jet)
    Colorbar(mosaic[2, 2], hm3)
    ax4 = Axis(mosaic[2, 3]; aspect = 1.0, title = L"$T_2$ [s]", xlabel = "x [μm]")
    hm4 = heatmap!(ax4, x, y, nferritin == 0 ? t2_myelin : t2_ferritin; colormap = :jet)
    Colorbar(mosaic[2, 4], hm4)
    plot && display(mosaic)

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
        filename = joinpath(savedir, string(timestamp, "_", basename, "-mosaic_", theta_suff, ext))
        FileIO.save(filename, mosaic)

        # Save data
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ".mat"))
        ndown = ngrid > 512 ? ngrid ÷ 512 : 1
        Idown = ndown:ndown:ngrid
        data = Dict{String, Any}()
        data["params"] = Dict(p)
        data["labels"] = regiondict()
        if nferritin == 0
            data["region_myelin"] = region_myelin[Idown, Idown]
            data["omega_myelin"] = ω_myelin[Idown, Idown]
            data["t1_myelin"] = t1_myelin[Idown, Idown]
            data["t2_myelin"] = t2_myelin[Idown, Idown]
        elseif nmyelin == 0
            data["region_ferritin"] = region_ferritin[Idown, Idown]
            data["omega_ferritin"] = ω_ferritin[Idown, Idown]
            data["t1_ferritin"] = t1_ferritin[Idown, Idown]
            data["t2_ferritin"] = t2_ferritin[Idown, Idown]
        else
            data["region_myelin"] = region_myelin[Idown, Idown]
            data["omega_myelin"] = ω_myelin[Idown, Idown]
            data["t1_myelin"] = t1_myelin[Idown, Idown]
            data["t2_myelin"] = t2_myelin[Idown, Idown]
            data["region_ferritin"] = region_ferritin[Idown, Idown]
            data["omega_ferritin"] = ω_ferritin[Idown, Idown]
            data["t1_ferritin"] = t1_ferritin[Idown, Idown]
            data["t2_ferritin"] = t2_ferritin[Idown, Idown]
            data["omega_total"] = ω_total[Idown, Idown]
        end
        matwrite(filename, data)
    end

    return nothing
end

function vessel_plot(; width = 1.0, ngrid = 4096, theta = 0.0, radius = 0.1, units = "rad/s", save = true, plot = true, ext = ".png")
    @assert ispow2(ngrid) "ngrid must be a power of 2"
    x = range(-width / 2, width / 2; length = ngrid)
    y = range(-width / 2, width / 2; length = ngrid)
    p = TissueParameters{Float64}(; theta = deg2rad(theta))

    blooddomain = BloodVesselDomain([Circle{2, Float64}(; centre = SA[0.0, 0.0], radius = radius)])
    ω_vessel = omegamap(x, y, p, blooddomain)
    t1_vessel = t1map(x, y, p, blooddomain)
    t2_vessel = t2map(x, y, p, blooddomain)
    region_vessel = regionmap(x, y, p, blooddomain)

    ω_bounds = maximum(abs, ω_vessel) .* (-1, 1) # extrema(ω_vessel)
    ω_rescale = lowercase(units) == "hz" ? 2π : 1.0
    ω_clamped = clamp.(ω_vessel, ω_bounds...)
    ω_label = string("Frequency [", lowercase(units) == "hz" ? "Hz" : "rad/s", "]")

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x [mm]", ylabel = "y [mm]", aspect = 1.0, title = L"Field Angle $\theta = %$(round(theta; digits = 1))\degree$")
    hm = heatmap!(ax, x, y, ω_clamped ./ ω_rescale; colormap = :jet, colorrange = ω_bounds ./ ω_rescale)
    Colorbar(fig[1, 2], hm; label = ω_label)
    plot && display(fig)

    mosaic = Figure()
    ax1 = Axis(mosaic[1, 1]; aspect = 1.0, title = L"%$(ω_label): $\theta = %$(round(theta; digits = 1))\degree$", ylabel = "y [mm]")
    hm1 = heatmap!(ax1, x, y, ω_clamped ./ ω_rescale; colormap = :jet, colorrange = ω_bounds ./ ω_rescale)
    Colorbar(mosaic[1, 2], hm1)
    ax2 = Axis(mosaic[1, 3]; aspect = 1.0, title = L"\text{Region}")
    hm2 = heatmap!(ax2, x, y, region_vessel; colormap = :jet, colorrange = (0, 4))
    Colorbar(mosaic[1, 4], hm2)
    ax3 = Axis(mosaic[2, 1]; aspect = 1.0, title = L"$T_1$ [s]", xlabel = "x [mm]", ylabel = "y [mm]")
    hm3 = heatmap!(ax3, x, y, t1_vessel; colormap = :jet)
    Colorbar(mosaic[2, 2], hm3)
    ax4 = Axis(mosaic[2, 3]; aspect = 1.0, title = L"$T_2$ [s]", xlabel = "x [mm]")
    hm4 = heatmap!(ax4, x, y, t2_vessel; colormap = :jet)
    Colorbar(mosaic[2, 4], hm4)
    plot && display(mosaic)

    if save === true
        timestamp = datestring()
        savedir = mkpath(joinpath(@__DIR__, "output"))
        basename = "vessel-field"
        theta_suff = string("theta-", theta == round(Int, theta) ? string(Int(theta)) : string(theta))

        # Save plots
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ext))
        FileIO.save(filename, fig)
        filename = joinpath(savedir, string(timestamp, "_", basename, "-mosaic_", theta_suff, ext))
        FileIO.save(filename, mosaic)

        # Save data
        filename = joinpath(savedir, string(timestamp, "_", basename, "_", theta_suff, ".mat"))
        ndown = ngrid > 512 ? ngrid ÷ 512 : 1
        Idown = ndown:ndown:ngrid
        data = Dict{String, Any}()
        data["params"] = Dict(p)
        data["labels"] = regiondict()
        data["region_vessel"] = region_vessel[Idown, Idown]
        data["omega_vessel"] = ω_vessel[Idown, Idown]
        data["t1_vessel"] = t1_vessel[Idown, Idown]
        data["t2_vessel"] = t2_vessel[Idown, Idown]
        matwrite(filename, data)
    end

    return nothing
end

function make_plots(; seed = 30, save = true, plot = true)
    # Vessel plot
    for theta in 0:10:90
        vessel_plot(; width = 1.0, theta, radius = 0.1, save, plot)
    end

    # Myelin plots
    for theta in 0:10:90
        Random.seed!(seed)
        myelin_plot(; width = 1.0, theta, nmyelin = 4, nferritin = 0, save, plot)
    end

    # Ferritin plot
    myelin_plot(; width = 0.5, theta = 90.0, nmyelin = 0, nferritin = 1, save, plot)

    return nothing
end
