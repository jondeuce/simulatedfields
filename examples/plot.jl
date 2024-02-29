using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MyelinFields
using MyelinFields: Circle, outer_circle, centre

using CairoMakie
using Dates
using Distributions
using FileIO
using MAT
using StaticArrays

set_theme!(theme_latexfonts())

datestring() = Dates.format(Dates.now(), "yyyy-mm-dd-T-HH-MM-SS-sss")

function rand_myelin_radii(p::BlochTorreyParameters, n...)
    return rand(Gamma(p.R_shape, p.R_scale), n...)
end

function rand_nonoverlapping_annulii(p::BlochTorreyParameters{T}; n = 10, width = 2.0, nattempts = 1000) where {T}
    annulii = Vector{Annulus{2, T}}(undef, n)
    for _ in 1:nattempts
        for i in 1:n
            centre = SA[rand(Uniform(-T(width) / 2, T(width) / 2)), rand(Uniform(-T(width) / 2, T(width) / 2))]
            radius = rand_myelin_radii(p)
            annulii[i] = Annulus(; centre, radius, g_ratio = p.g_ratio)
        end
        any(isoverlapping(annulii[i], annulii[j]) for i in 1:n for j in i+1:n) && continue
        density = mean(1:10_000) do i
            x = SA[rand(Uniform(-T(width) / 2, T(width) / 2)), rand(Uniform(-T(width) / 2, T(width) / 2))]
            return any(x ∈ MyelinFields.outer_circle(annulus) for annulus in annulii)
        end
        density < 0.65 && continue
        return annulii
    end
    return error("Failed to generate non-overlapping annulii after $nattempts attempts")
end

function rand_ferritin_spheres(p::BlochTorreyParameters{T}; n = 10, width = 2.0, nattempts = 1000) where {T}
    spheres = Vector{Circle{3, T}}(undef, n)
    for _ in 1:nattempts
        for i in 1:n
            centre = T(0.95) .* SA[rand(Uniform(-T(width) / 2, T(width) / 2)), rand(Uniform(-T(width) / 2, T(width) / 2)), zero(T)]
            radius = p.R_Ferritin
            spheres[i] = Circle(; centre, radius)
        end
        minimum(norm(centre(spheres[i]) - centre(spheres[j])) for i in 1:n for j in i+1:n) > width / 4 && break
    end
    return spheres
end

function make_plot(; width = 1.2, ngrid = 4096, nmyelin = 5, nferritin = 3, nattempts = 100_000_000, units = "Hz", save = true, ext = ".png")
    @assert ispow2(ngrid) "ngrid must be a power of 2"
    x = range(-width / 2, width / 2; length = ngrid)
    y = range(-width / 2, width / 2; length = ngrid)
    p = BlochTorreyParameters{Float64}()

    annulii = rand_nonoverlapping_annulii(p; n = nmyelin, width = width, nattempts = nattempts)
    spheres = rand_ferritin_spheres(p; n = nferritin, width = width)

    # Note: could use `omega(x, y, p, annulii, spheres)` directly, but this way we get the bounds for the colorbar,
    # otherwise the ferritin fields will dominate and you won't see the myelin fields
    ω_myelin = omega(x, y, p, annulii, Circle{3, Float64}[])
    ω_ferritin = omega(x, y, p, Annulus{2, Float64}[], spheres)
    ω_total = ω_myelin .+ ω_ferritin

    ω_myelin_bounds = maximum(abs, ω_myelin) .* (-1, 1) # extrema(ω_myelin)
    ω_clamped = clamp.(ω_total, ω_myelin_bounds...)

    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "x [μm]", ylabel = "y [μm]", aspect = 1.0)
    hm = heatmap!(ax, x, y, lowercase(units) == "hz" ? ω_clamped ./ 2π : ω_clamped; colormap = :jet)
    Colorbar(fig[1, 2], hm; label = lowercase(units) == "hz" ? "Frequency [Hz]" : "ω [rad/s]")

    if save === true || save === :prompt
        if save === :prompt
            display(fig)
            println("Save plot and data? [y/N]")
            lowercase(readline()) != "y" && return fig
        end
        timestamp = datestring()
        savedir = mkpath(joinpath(@__DIR__, "output"))

        # Save plot
        filename = joinpath(savedir, string(timestamp, "_myelin-field-with-ferritin", ext))
        FileIO.save(filename, fig)

        # Save data
        filename = joinpath(savedir, string(timestamp, "_myelin-field-with-ferritin.mat"))
        ndown = ngrid > 512 ? ngrid ÷ 512 : 1
        Idown = ndown:ndown:ngrid
        data = Dict{String, Any}(
            "params" => Dict(p),
            "omega_myelin" => ω_myelin[Idown, Idown],
            "omega_ferritin" => ω_ferritin[Idown, Idown],
            "omega_total" => ω_total[Idown, Idown],
        )
        matwrite(filename, data)
    end

    return fig
end
