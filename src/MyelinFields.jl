module MyelinFields

using Parameters: @with_kw
using StaticArrays: SVector
using LinearAlgebra: ⋅, dot, norm

export BlochTorreyParameters, MyelinDomain
export Annulus, Circle, isoverlapping
export omega

# ---------------------------------------------------------------------------- #
# BlochTorreyParameters
#   Physical parameters needed for Bloch-Torrey simulations
# ---------------------------------------------------------------------------- #
@with_kw struct BlochTorreyParameters{T <: AbstractFloat}
    B0::T              = T(3.0) # ............................................ [T]          External magnetic field (z-direction)
    gamma::T           = T(2.67515255e8) # ................................... [rad/s/T]    Gyromagnetic ratio
    theta::T           = T(π) / 2 # .......................................... [rad]        Main magnetic field angle w.r.t B0
    R1_sp::T           = T(inv(200e-3)) # .................................... [1/s]        1/T1 relaxation rate of small pool (myelin) water [REF: TODO]
    R1_lp::T           = T(inv(1084e-3)) # ................................... [1/s]        1/T1 relaxation rate of large pool (intra-cellular/axonal) water [REF: TODO]
    R1_Tissue::T       = T(inv(1084e-3)) # ................................... [1/s]        1/T1 relaxation rate of white matter tissue (extra-cellular water) [REF: 1084 ± 45; https://www.ncbi.nlm.nih.gov/pubmed/16086319]
    R2_sp::T           = T(inv(15e-3)) # ..................................... [1/s]        1/T2 relaxation rate of small pool (myelin) water (Xu et al. 2017)
    R2_lp::T           = T(inv(63e-3)) # ..................................... [1/s]        1/T2 relaxation rate of large pool (intra-cellular/axonal) water [REF: TODO]
    R2_lp_DipDip::T    = T(0.0) # ............................................ [1/s]        1/T2 dipole-dipole interaction coefficient for large pool (intra-cellular/axonal) water: R2_lp(θ) = R2_lp + R2_lp_DipDip * (3*cos^2(θ)−1)^2 [REF: https://doi.org/10.1016/j.neuroimage.2013.01.051]
    R2_Tissue::T       = T(inv(63e-3)) # ..................................... [1/s]        1/T2 relaxation rate of white matter tissue (extra-cellular water) (was 14.5Hz; changed to match R2_lp) [REF: TODO]
    R2_Water::T        = T(inv(2.2)) # ....................................... [1/s]        Relaxation rate of pure water
    D_Water::T         = T(3037.0) # ......................................... [um^2/s]     Diffusion coefficient in water
    D_Blood::T         = T(3037.0) # ......................................... [um^2/s]     Diffusion coefficient in blood
    D_Tissue::T        = T(1000.0) # ......................................... [um^2/s]     TODO (reference?) Diffusion coefficient in tissue
    D_Sheath::T        = T(1000.0) # ......................................... [um^2/s]     TODO (reference?) Mean diffusivity coefficient in myelin sheath
    D_Axon::T          = T(1000.0) # ......................................... [um^2/s]     TODO (reference?) Diffusion coefficient in axon interior
    FRD_Sheath::T      = T(0.5) # ............................................ [unitless]   TODO (reference?) Fractional radial diffusivity within myelin sheath; FRD ∈ [0,1] where 0 is purely polar, 0.5 is isotropic, and 1 is purely radial
    K_perm::T          = T(1.0e-3) # ......................................... [um/s]       TODO (reference?) Interface permeability constant
    K_Axon_Sheath::T   = T(K_perm) # ......................................... [um/s]       Axon-Myelin interface permeability
    K_Tissue_Sheath::T = T(K_perm) # ......................................... [um/s]       Tissue-Myelin interface permeability
    R_mu::T            = T(0.46) # ........................................... [um]         Axon mean radius (taken to be outer radius)
    R_shape::T         = T(5.7) # ............................................ [um/um]      Axon shape parameter for Gamma distribution (Xu et al. 2017)
    R_scale::T         = T(R_mu / R_shape) # ................................. [um]         Axon scale parameter for Gamma distribution (Xu et al. 2017)
    PD_sp::T           = T(0.5) # ............................................ [unitless]   Relative proton density (Myelin)
    PD_lp::T           = T(1.0) # ............................................ [unitless]   Relative proton density (Intra Extra)
    PD_Fe::T           = T(1.0) # ............................................ [unitless]   Relative proton density (Ferritin)
    g_ratio::T         = T(0.8370) # ......................................... [um/um]      g-ratio (originally 0.71; 0.84658 for healthy, 0.8595 for MS)
    AxonPDensity::T    = T(0.83) # ........................................... [um^2/um^2]  Axon packing density based region in white matter (Xu et al. 2017) (originally 0.83)
    MVF::T             = T(AxonPDensity * (1 - g_ratio^2)) # ................. [um^3/um^3]  Myelin volume fraction, assuming periodic circle packing and constant g_ratio
    MWF::T             = T(PD_sp * MVF / (PD_lp - (PD_lp - PD_sp) * MVF)) # .. [unitless]   Myelin water fraction, assuming periodic circle packing and constant g_ratio
    MyelinChiI::T      = T(-60e-9) # ......................................... [T/T]        Isotropic susceptibility of myelin; Wharton and Bowtell 2012 find -60 ppb ± 20 ppb in Table 2, Xu et al. 2017 use the same value.
    MyelinChiA::T      = T(-140e-9) # ........................................ [T/T]        Anisotropic Susceptibility of myelin; Wharton and Bowtell 2012 find -140 ppb ± 20 ppb in Table 2, Xu et al. 2017 use a slightly different value of -120 ppb.
    MyelinChiE::T      = T(0.0) # ............................................ [T/T]        Exchange component to resonance freqeuency; we default to 0.0, but Wharton and Bowtell 2012 find 20 ppb ± 10 ppb and 50 ppb ± 10 ppb in Table 2.
    R2_Fe::T           = T(inv(1e-6)) # ...................................... [1/s]        Relaxation rate of iron in ferritin (assumed extremely high)
    R_Ferritin::T      = T(4.0e-3) # ......................................... [um]         Ferritin mean radius
    Fe_Conc::T         = T(0.0424) # ......................................... [g/g]        TODO (check units) Concentration of iron in the frontal white matter (0.0424 in frontal WM; 0.2130 in globus pallidus deep grey matter)
    Rho_Tissue::T      = T(1.073) # .......................................... [g/ml]       White matter tissue density
    Chi_Tissue::T      = T(-9.05e-6) # ....................................... [T/T]        Isotropic susceptibility of tissue
    Chi_FeUnit::T      = T(1.4e-9) # ......................................... [g/g]        TODO (check units) Susceptibility of iron per ppm/(ug/g) weight fraction of iron.
    Chi_FeFull::T      = T(520.0e-6) # ....................................... [T/T]        Susceptibility of iron for ferritin particle FULLY loaded with 4500 iron atoms. (use volume of FULL spheres) (from Contributions to magnetic susceptibility)
    Rho_Fe::T          = T(7.874) # .......................................... [g/cm^3]     Iron density

    # Misc checks
    @assert K_perm ≈ K_Axon_Sheath   # different permeabilities not implemented
    @assert K_perm ≈ K_Tissue_Sheath # different permeabilities not implemented
    @assert R_scale ≈ R_mu / R_shape
    @assert MVF ≈ AxonPDensity * (1 - g_ratio^2)
    @assert MWF ≈ PD_sp * MVF / (PD_lp - (PD_lp - PD_sp) * MVF)
end

Base.Dict(p::BlochTorreyParameters{T}) where {T} = Dict{String, T}(string(k) => getfield(p, k) for k in fieldnames(typeof(p)))

@with_kw struct Circle{dim, T}
    centre::SVector{dim, T}
    radius::T
end
@inline centre(c::Circle) = c.centre
@inline radius(c::Circle) = c.radius

@inline function Base.in(x::SVector{dim, T}, circle::Circle{dim, T}, thresh::T = zero(T)) where {dim, T}
    dx = x - centre(circle)
    return dx ⋅ dx <= (radius(circle) + thresh)^2
end

@with_kw struct Annulus{dim, T}
    centre::SVector{dim, T}
    radius::T
    g_ratio::T
end
@inline centre(a::Annulus) = a.centre
@inline inner_radius(a::Annulus) = a.radius * a.g_ratio
@inline outer_radius(a::Annulus) = a.radius
@inline inner_circle(a::Annulus) = Circle(centre(a), inner_radius(a))
@inline outer_circle(a::Annulus) = Circle(centre(a), outer_radius(a))
@inline radii(a::Annulus) = (inner_radius(a), outer_radius(a))

@inline function Base.in(x::SVector{dim, T}, annulus::Annulus{dim, T}, thresh::T = zero(T)) where {dim, T}
    ri, ro = radii(annulus)
    dx = x - centre(annulus)
    return (ri - thresh)^2 <= dx ⋅ dx <= (ro + thresh)^2
end

@inline function isoverlapping(a::Annulus{dim, T}, b::Annulus{dim, T}, thresh::T = zero(T)) where {dim, T}
    dx = centre(a) - centre(b)
    return dx ⋅ dx <= (outer_radius(a) + outer_radius(b) - thresh)^2
end

struct MyelinDomain{T}
    annulii::Vector{Annulus{2, T}}
    ferritins::Vector{Circle{3, T}}
end

@enum Region TissueRegion MyelinRegion AxonRegion

# ---------------------------------------------------------------------------- #
# Local frequency perturbation map functions
# ---------------------------------------------------------------------------- #
struct OmegaDerivedConstants{T}
    ω₀::T
    s²::T
    c²::T
    function OmegaDerivedConstants(p::BlochTorreyParameters{T}) where {T}
        γ, B₀, θ = p.gamma, p.B0, p.theta
        ω₀ = γ * B₀
        s, c = sincos(θ)
        return new{T}(ω₀, s^2, c^2)
    end
end

@inline function omega_myelin_tissue(x::SVector{2, T}, p::BlochTorreyParameters{T}, b::OmegaDerivedConstants{T}, c_in::Circle{2, T}, c_out::Circle{2, T}) where {T}
    χI, χA, ri², ro² = p.MyelinChiI, p.MyelinChiA, radius(c_in)^2, radius(c_out)^2
    dx = x - centre(c_in)
    r² = dx ⋅ dx
    cos2ϕ = (dx[1] - dx[2]) * (dx[1] + dx[2]) / r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²
    tmp = b.s² * cos2ϕ * ((ro² - ri²) / r²) # Common calculation
    I = χI / 2 * tmp # isotropic component
    A = χA / 8 * tmp # anisotropic component
    return b.ω₀ * (I + A)
end
@inline omega_myelin_tissue(x::SVector{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, a::Annulus{2}) = omega_myelin_tissue(x, p, b, inner_circle(a), outer_circle(a))

@inline function omega_myelin(x::SVector{2, T}, p::BlochTorreyParameters{T}, b::OmegaDerivedConstants{T}, c_in::Circle{2, T}, c_out::Circle{2, T}) where {T}
    χI, χA, E, ri², ro = p.MyelinChiI, p.MyelinChiA, p.MyelinChiE, radius(c_in)^2, radius(c_out)
    dx = x - centre(c_in)
    r² = dx ⋅ dx
    cos2ϕ = (dx[1] - dx[2]) * (dx[1] + dx[2]) / r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²
    r = √r²
    I = χI * (b.c² - T(1) / 3 - b.s² * cos2ϕ * ri² / r²) / 2 # isotropic component
    A = χA * (b.s² * (-T(5) / 12 - cos2ϕ / 8 * (1 + ri² / r²) + T(3) / 4 * log(ro / r)) - b.c² / 6) # anisotropic component
    return b.ω₀ * (I + A + E)
end
@inline omega_myelin(x::SVector{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, a::Annulus{2}) = omega_myelin(x, p, b, inner_circle(a), outer_circle(a))

@inline function omega_myelin_axon(x::SVector{2, T}, p::BlochTorreyParameters{T}, b::OmegaDerivedConstants{T}, c_in::Circle{2, T}, c_out::Circle{2, T}) where {T}
    χA, ri, ro = p.MyelinChiA, radius(c_in), radius(c_out)
    A = 3χA / 4 * b.s² * log(ro / ri) # anisotropic (and only) component
    return b.ω₀ * A
end
@inline omega_myelin_axon(x::SVector{2}, p::BlochTorreyParameters, b::OmegaDerivedConstants, a::Annulus{2}) = omega_myelin_axon(x, p, b, inner_circle(a), outer_circle(a))

@inline function omega_ferritin_outside(x::SVector{3, T}, p::BlochTorreyParameters{T}, b::OmegaDerivedConstants{T}, c::Circle{3, T}) where {T}
    @assert p.theta ≈ T(π) / 2 "Ferritin spheres not yet implemented with θ != π/2; got θ = $(p.theta)" #TODO
    χ, a = p.Chi_FeFull, radius(c)
    dx = x - centre(c)
    r = norm(dx) # 3D radius: r = √(x²+y²+z²)
    r²xy = dx[1]^2 + dx[2]^2 # in-plane 2D radius
    cos²ϕ = ifelse(r²xy == 0, zero(T), dx[1]^2 / r²xy) # azimuthal angle: cosϕ = x/√(x²+y²) (note: limit as r²xy → 0 is undefined, can be any number in [0,1] depending how you take the limit; arbitrarily choose 0)
    A = (χ / 3) * (3 * cos²ϕ - 1) * (a / r)^3 # field outside a sphere of constant susceptibility (Cheng Y-CN, Neelavalli J, Haacke EM. Limitations of Calculating Field Distributions and Magnetic Susceptibilities in MRI using a Fourier Based Method. Phys Med Biol 2009; 54: 1169–1189)
    return b.ω₀ * A
end

@inline function omega_ferritin_inside(x::SVector{3, T}, p::BlochTorreyParameters{T}, b::OmegaDerivedConstants{T}, c::Circle{3, T}) where {T}
    return zero(T) # no field inside a sphere of constant susceptibility
end

# ---------------------------------------------------------------------------- #
# Global frequency perturbation functions: calculate ω(x) due to entire domain
# ---------------------------------------------------------------------------- #

# Calculate ω(x) by searching for the region which `x` is contained in
function omega(
    x::SVector{2, T},
    p::BlochTorreyParameters{T},
    annulii::Vector{Annulus{2, T}},
    spheres::Vector{Circle{3, T}},
) where {T}

    !isempty(spheres) && !(p.theta ≈ T(π) / 2) && error("Ferritin spheres not yet implemented with θ != π/2; got θ = $(p.theta)")

    # If there are no structures, then there is no frequency shift ω
    isempty(annulii) && isempty(spheres) && return zero(eltype(x))

    constants = OmegaDerivedConstants(p)
    ω = zero(T)

    # Find the region that `x` is in
    i_outer = findfirst(c -> x ∈ outer_circle(c), annulii)
    i_inner = findfirst(c -> x ∈ inner_circle(c), annulii)

    region = if i_outer === nothing
        @assert i_inner === nothing "Point is outside outer circles but inside an inner circle"
        TissueRegion # not in outer circles -> tissue region
    elseif i_inner !== nothing
        @assert i_outer == i_inner "Inner and outer circles are not corresponding"
        AxonRegion # in inner circles -> axon region
    else
        MyelinRegion # in outer circles but not inner circles -> myelin region
    end

    # Add contributions from myelin sheaths
    if region == TissueRegion
        @inbounds for i in eachindex(annulii)
            ω += omega_myelin_tissue(x, p, constants, annulii[i])
        end
    elseif region == MyelinRegion
        @inbounds for i in eachindex(annulii)
            ω += i == i_outer ?
                 omega_myelin(x, p, constants, annulii[i]) :
                 omega_myelin_tissue(x, p, constants, annulii[i])
        end
    else # region == AxonRegion
        @inbounds for i in eachindex(annulii)
            ω += i == i_outer ?
                 omega_myelin_axon(x, p, constants, annulii[i]) :
                 omega_myelin_tissue(x, p, constants, annulii[i])
        end
    end

    # Add contributions from ferritin spheres
    @inbounds for i in eachindex(spheres)
        x3D = SVector{3, T}(x[1], x[2], zero(T)) # pad to 3D
        ω += x3D ∈ spheres[i] ?
             omega_ferritin_inside(x3D, p, constants, spheres[i]) :
             omega_ferritin_outside(x3D, p, constants, spheres[i])
    end

    return ω
end

# Individual coordinate input
omega(x::T, y::T, p::BlochTorreyParameters{T}, annulii::Vector{Annulus{2, T}}, spheres::Vector{Circle{3, T}}) where {T} = omega(SVector{2, T}((x, y)), p, annulii, spheres)
omega(x::T, y::T, p::BlochTorreyParameters{T}, domain::MyelinDomain{T}) where {T} = omega(x, y, p, domain.annulii, domain.ferritins)

function omega(x::AbstractVector{T}, y::AbstractVector{T}, p::BlochTorreyParameters{T}, args...) where {T}
    ω = zeros(T, length(x), length(y))
    Threads.@threads for j in 1:length(y)
        for i in 1:length(x)
            ω[i, j] = omega(x[i], y[j], p, args...)
        end
    end
    return ω
end

end # module MyelinFields
