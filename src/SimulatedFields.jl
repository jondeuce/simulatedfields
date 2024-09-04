module SimulatedFields

using Parameters: @with_kw
using StaticArrays: SA, SVector
using LinearAlgebra: ⋅, dot, norm, normalize

export BloodVesselDomain, MyelinDomain, TissueParameters
export Region, TissueRegion, BloodRegion, MyelinRegion, AxonRegion, FerritinRegion
export Annulus, Circle, isoverlapping
export omegamap, t1map, t2map, regionmap, mapdomain, findregion, regiondict

####
#### TissueParameters
####    Physical parameters for calculating frequency maps, relataion maps, etc.
####

@with_kw struct TissueParameters{T <: AbstractFloat}
    B0::T              = T(3.0) # ............................................ [T]          External magnetic field (z-direction)
    gamma::T           = T(2.67515255e8) # ................................... [rad/s/T]    Gyromagnetic ratio
    theta::T           = T(π) / 2 # .......................................... [rad]        Main magnetic field angle w.r.t B0
    R1_sp::T           = T(inv(115.6e-3)) # .................................. [1/s]        1/T1 relaxation rate of small pool (myelin) water [REF: mean of measurements with B0 = 3T, TIₙ = 8s in Table 2 of Labadie C, Lee J-H, Rooney WD, et al. Myelin water mapping by spatially regularized longitudinal relaxographic imaging at high magnetic fields. Magnetic Resonance in Medicine 2014; 71: 375–387.]
    R1_lp::T           = T(inv(1084e-3)) # ................................... [1/s]        1/T1 relaxation rate of large pool (intra-cellular/axonal) water [REF: 1084 ± 45; https://www.ncbi.nlm.nih.gov/pubmed/16086319]
    R1_Tissue::T       = T(R1_lp) # .......................................... [1/s]        1/T1 relaxation rate of white matter tissue (extra-cellular water)
    R1_Blood::T        = T(inv(1649e-3)) # ....................................[1/s]        1/T1 relaxation rate of blood [REF: 1649 ± 68; https://doi.org/10.1002/mrm.24550]
    R2_sp::T           = T(inv(15e-3)) # ..................................... [1/s]        1/T2 relaxation rate of small pool (myelin) water (Xu et al. 2017)
    R2_lp::T           = T(inv(63e-3)) # ..................................... [1/s]        1/T2 relaxation rate of large pool (intra-cellular/axonal) water [REF: TODO]
    R2_lp_DipDip::T    = T(0.0) # ............................................ [1/s]        1/T2 dipole-dipole interaction coefficient for large pool (intra-cellular/axonal) water: R2_lp(θ) = R2_lp + R2_lp_DipDip * (3*cos^2(θ)−1)^2 [REF: https://doi.org/10.1016/j.neuroimage.2013.01.051]
    R2_Tissue::T       = T(inv(63e-3)) # ..................................... [1/s]        1/T2 relaxation rate of white matter tissue (extra-cellular water) (was 14.5Hz; changed to match R2_lp) [REF: TODO]
    R2_Water::T        = T(inv(2200e-3)) # ................................... [1/s]        Relaxation rate of pure water
    R2_CSF::T          = T(inv(1790e-3)) # ................................... [1/s]        Relaxation rate of cerebrospinal fluid at 3T [ms]; value corrected for partial volume effects is 1790 ms; uncorrected value is 1672 ms [REF: Spijkerman J, Petersen E, Hendrikse J, et al. T2 mapping of cerebrospinal fluid: 3T versus 7T.]
    D_Water::T         = T(3037.0) # ......................................... [um^2/s]     Diffusion coefficient in water
    D_Blood::T         = T(3037.0) # ......................................... [um^2/s]     Diffusion coefficient in blood
    D_Tissue::T        = T(1000.0) # ......................................... [um^2/s]     TODO: (reference?) Diffusion coefficient in tissue
    D_Sheath::T        = T(1000.0) # ......................................... [um^2/s]     TODO: (reference?) Mean diffusivity coefficient in myelin sheath
    D_Axon::T          = T(1000.0) # ......................................... [um^2/s]     TODO: (reference?) Diffusion coefficient in axon interior
    FRD_Sheath::T      = T(0.5) # ............................................ [fraction]   TODO: (reference?) Fractional radial diffusivity within myelin sheath; FRD ∈ [0,1] where 0 is purely polar, 0.5 is isotropic, and 1 is purely radial
    K_perm::T          = T(1.0e-3) # ......................................... [um/s]       TODO: (reference?) Interface permeability constant
    K_Axon_Sheath::T   = T(K_perm) # ......................................... [um/s]       Axon-Myelin interface permeability
    K_Tissue_Sheath::T = T(K_perm) # ......................................... [um/s]       Tissue-Myelin interface permeability
    R_mu::T            = T(0.46) # ........................................... [um]         Axon mean radius (taken to be outer radius)
    R_shape::T         = T(5.7) # ............................................ [um/um]      Axon shape parameter for Gamma distribution (Xu et al. 2017)
    R_scale::T         = T(R_mu / R_shape) # ................................. [um]         Axon scale parameter for Gamma distribution (Xu et al. 2017)
    PD_sp::T           = T(0.5) # ............................................ [fraction]   Relative proton density (Myelin)
    PD_lp::T           = T(1.0) # ............................................ [fraction]   Relative proton density (Intra Extra)
    PD_Fe::T           = T(1.0) # ............................................ [fraction]   Relative proton density (Ferritin)
    g_ratio::T         = T(0.7) # ............................................ [um/um]      g-ratio (see discussion in Stikov N, Campbell JSW, Stroh T, et al. In vivo histology of the myelin g-ratio with magnetic resonance imaging. NeuroImage 2015; 118: 397–405) (previously used 0.837; 0.84658 for healthy, 0.8595 for MS)
    axon_density::T    = T(0.83) # ........................................... [um^2/um^2]  Axon packing density based region in white matter (Xu et al. 2017)
    MVF::T             = T(axon_density * (1 - g_ratio^2)) # ................. [um^3/um^3]  Myelin volume fraction, assuming periodic circle packing and constant g_ratio
    MWF::T             = T(PD_sp * MVF / (PD_lp - (PD_lp - PD_sp) * MVF)) # .. [fraction]   Myelin water fraction, assuming periodic circle packing and constant g_ratio
    MyelinChiI::T      = T(-60e-9) # ......................................... [T/T]        Isotropic susceptibility of myelin; Wharton and Bowtell 2012 find -60 ppb ± 20 ppb in Table 2, Xu et al. 2017 use the same value.
    MyelinChiA::T      = T(-140e-9) # ........................................ [T/T]        Anisotropic Susceptibility of myelin; Wharton and Bowtell 2012 find -140 ppb ± 20 ppb in Table 2, Xu et al. 2017 use a slightly different value of -120 ppb.
    MyelinChiE::T      = T(0.0) # ............................................ [T/T]        Exchange component to resonance freqeuency; we default to 0.0, but Wharton and Bowtell 2012 find 20 ppb ± 10 ppb and 50 ppb ± 10 ppb in Table 2.
    R2_Fe::T           = T(inv(1e-6)) # ...................................... [1/s]        R2 relation rate of iron in ferritin (assumed extremely high) #TODO Ref
    R1_Fe::T           = T(inv(1e-6)) # ...................................... [1/s]        R1 relation rate of iron in ferritin (assumed extremely high) #TODO Ref
    R_Ferritin::T      = T(4.0e-3) # ......................................... [um]         Ferritin mean radius (Harrison PM, Arosio P. The ferritins: molecular properties, iron storage function and cellular regulation. Biochimica et Biophysica Acta (BBA) - Bioenergetics 1996; 1275: 161–203.)
    Fe_Conc::T         = T(0.0424) # ......................................... [g/g]        TODO: (check units) Concentration of iron in the frontal white matter (0.0424 in frontal WM; 0.2130 in globus pallidus deep grey matter)
    Rho_Tissue::T      = T(1.073) # .......................................... [g/ml]       White matter tissue density
    Chi_Tissue::T      = T(-9.05e-6) # ....................................... [T/T]        Isotropic susceptibility of tissue
    Chi_FeUnit::T      = T(1.4e-9) # ......................................... [g/g]        TODO: (check units) Susceptibility of iron per ppm/(ug/g) weight fraction of iron.
    Chi_FeFull::T      = T(520.0e-6) # ....................................... [T/T]        Susceptibility of iron for ferritin particle FULLY loaded with 4500 iron atoms; use volume of FULL spheres. (Contributions to magnetic susceptibility of brain tissue - Duyn - 2017 - NMR in Biomedicine - Wiley Online Library, https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/nbm.3546)
    Rho_Fe::T          = T(7.874) # .......................................... [g/cm^3]     Iron density
    Hct_Blood::T       = T(0.44) # ........................................... [fraction]   Hematocrit fraction
    Yv_Blood::T        = T(0.61) # ........................................... [fraction]   Venous Blood Oxygenation
    Ya_Blood::T        = T(0.98) # ........................................... [fraction]   Arterial Blood Oxygenation [REF: Zhao et al., 2007, MRM, Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T]
    dR2v_Blood::T      = T(11 + 125 * (1 - Yv_Blood)^2) # .................... [1/s]        Increase in R2 in venous blood relative to tissue [REF: Zhao et al., 2007]
    dR2a_Blood::T      = T(11 + 125 * (1 - Ya_Blood)^2) # .................... [1/s]        Increase in R2 in arterial blood relative to tissue [REF: Zhao et al., 2007]
    dChiv_Blood::T     = T(2.26e-6 * Hct_Blood * (1 - Yv_Blood)) # ........... [T/T]        Increase in susceptibility in venous blood relative to tissue
    dChia_Blood::T     = T(2.26e-6 * Hct_Blood * (1 - Ya_Blood)) # ........... [T/T]        Increase in susceptibility in arterial blood relative to tissue
    CA_Blood::T        = T(0.0) # ............................................ [mM]         Contrast Agent concentration
    dR2_CA_Blood::T    = T(5.2) # ............................................ [Hz/mM]      Change in R2 per mM of contrast agent
    dChi_CA_Blood::T   = T(0.3393e-6) # ...................................... [T/T/mM]     Change in susceptibility per mM of contrast agent
    R2v_Blood::T       = T(dR2v_Blood + CA_Blood * dR2_CA_Blood) # ........... [1/s]        R2 in venous blood
    R2a_Blood::T       = T(dR2a_Blood + CA_Blood * dR2_CA_Blood) # ........... [1/s]        R2 in arterial blood
    Chiv_Blood::T      = T(dChiv_Blood + CA_Blood * dChi_CA_Blood) # ......... [T/T]        Susceptibility in venous blood relative to tissue
    Chia_Blood::T      = T(dChia_Blood + CA_Blood * dChi_CA_Blood) # ......... [T/T]        Susceptibility in arterial blood relative to tissue

    # Constraints on parameters
    @assert B0 == 3.0 "Tissue parameters are only valid for B0 = 3.0 T"
    @assert Hct_Blood == 0.44 "Tissue parameters are only valid for Hct_Blood = 0.44"

    # Misc checks
    @assert K_perm ≈ K_Axon_Sheath   # different permeabilities not implemented
    @assert K_perm ≈ K_Tissue_Sheath # different permeabilities not implemented
    @assert R_scale ≈ R_mu / R_shape
    @assert MVF ≈ axon_density * (1 - g_ratio^2)
    @assert MWF ≈ PD_sp * MVF / (PD_lp - (PD_lp - PD_sp) * MVF)
end

Base.Dict(p::TissueParameters{T}) where {T} = Dict{String, T}(string(k) => getfield(p, k) for k in fieldnames(typeof(p)))

####
#### Geometry utils
####

norm2(x::SVector{dim, T}) where {dim, T} = x ⋅ x
liftdim(x::SVector{dim, T}) where {dim, T} = SVector{dim + 1, T}(x..., zero(T))

@with_kw struct Circle{dim, T}
    centre::SVector{dim, T}
    radius::T
end
@inline centre(c::Circle) = c.centre
@inline radius(c::Circle) = c.radius

@inline function Base.in(x::SVector{dim, T}, circle::Circle{dim, T}, thresh::T = zero(T)) where {dim, T}
    dx = x - centre(circle)
    return norm2(dx) <= (radius(circle) + thresh)^2
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
    return (ri - thresh)^2 <= norm2(dx) <= (ro + thresh)^2
end

@inline function isoverlapping(a::Annulus{dim, T}, b::Annulus{dim, T}, thresh::T = zero(T)) where {dim, T}
    dx = centre(a) - centre(b)
    return norm2(dx) <= (outer_radius(a) + outer_radius(b) - thresh)^2
end

function area_weighted_mean(circles::Vector{Circle{2, T}}) where {T}
    μ = SA[zero(T), zero(T)]
    Σr² = zero(T)
    for c in circles
        Σr² += radius(c)^2
        μ += centre(c) * radius(c)^2
    end
    return μ / Σr²
end

####
#### Greedy circle packing
####

function solve_triangle(a::T, b::T, c::T) where {T}
    # Given three prescribed sidelengths of a triangle (a, b, c) and two prescribed points P1 = (0, 0) and P2 = (a, 0),
    # return the two solutions for the third point P3 = (x, y) and P4 = (x, -y).
    x = (a^2 - b^2 + c^2) / (2 * a)
    y = √((-a + b + c) * (a - b + c) * (a + b - c) * (a + b + c)) / (2 * a)
    return SA[x, y], SA[x, -y]
end

function tangent_circles(c1::Circle{2, T}, c2::Circle{2, T}, new_radius::T, min_separation::T = zero(T)) where {T}
    # Given two circles `c1` and `c2`, return the two circles that are tangent to both `c1` and `c2` (up to `min_separation` distance).
    dx = centre(c2) - centre(c1)
    d12 = norm(dx)
    d23 = radius(c2) + new_radius + min_separation
    d32 = new_radius + radius(c1) + min_separation

    # Compute new tangent circles `c1` and `c2` are close enough, otherwise return `nothing`
    d12 >= d23 + d32 && return nothing
    new_rel_centre1, new_rel_centre2 = solve_triangle(d12, d23, d32)

    # Affine transformation to rotate and translate the new circles to the correct position
    cosϕ, sinϕ = dx[1] / d12, dx[2] / d12
    R = SA[cosϕ -sinϕ; sinϕ cosϕ]
    new_centre1 = centre(c1) + R * new_rel_centre1
    new_centre2 = centre(c1) + R * new_rel_centre2

    return Circle(new_centre1, new_radius), Circle(new_centre2, new_radius)
end

function packing_energy(circles::Vector{Circle{2, T}}, new_circle::Circle{2, T}) where {T}
    # Packing energy is sum of squared distances from each circle centre
    energy = zero(T)
    x⃗ = centre(new_circle)
    for c in circles
        dx = x⃗ - centre(c)
        energy += norm2(dx)
    end
    return energy
end

function signed_overlap_distance(c1::Circle{2, T}, c2::Circle{2, T}, min_separation::T = zero(T)) where {T}
    dx = centre(c1) - centre(c2)
    return norm2(dx) - (radius(c1) + radius(c2) + min_separation)^2
end

#=
using NLopt: NLopt
using ForwardDiff: ForwardDiff, DiffResults

function greedy_insert_circle_opt(
    circles::Vector{Circle{2, Float64}},
    new_radius::Float64,
    min_separation::Float64 = 0.0,
)

    function packing_energy_opt(x::Vector{Float64})
        # Packing energy is sum of squared distances from each circle centre
        new_circle = Circle(SA[x[1], x[2]], new_radius)
        return packing_energy(circles, new_circle)
    end

    function signed_overlap_constraint(x::Vector{Float64}, c::Circle{2, Float64})
        # Constrain the new circle to be at least `min_separation` distance from the circle `c`
        new_circle = Circle(SA[x[1], x[2]], new_radius)
        return -signed_overlap_distance(new_circle, c, min_separation) # constraint is *negative* when satisfied
    end

    function nlopt_grad(f, x::Vector{Float64}, grad::Vector{Float64})
        if length(grad) > 0
            res = DiffResults.DiffResult(0.0, grad)
            ForwardDiff.gradient!(res, f, x)
            return DiffResults.value(res)
        else
            return f(x)
        end
    end

    optfun(x::Vector, grad::Vector) = nlopt_grad(packing_energy, x, grad)
    constraintfun(x::Vector, grad::Vector, c::Circle) = nlopt_grad(y -> signed_overlap_constraint(y, c), x, grad)

    opt = NLopt.Opt(:LD_MMA, 2)
    opt.xtol_rel = 1e-4
    opt.maxeval = 100
    opt.min_objective = optfun
    for c in circles
        NLopt.inequality_constraint!(opt, (x, g) -> constraintfun(x, g, c), 0.0)
    end

    while true
        μ0 = Vector(area_weighted_mean(circles)) # mean of existing centres
        r0 = sum([radius.(circles); new_radius] .+ min_separation / 2) # enclosing radius
        x0 = μ0 + 1 * r0 * normalize(randn(2)) # initial guess
        (minf, minx, ret) = NLopt.optimize(opt, x0)

        ret === :MAXEVAL_REACHED && continue
        return Circle(SA[minx[1], minx[2]], new_radius)
    end
end
=#

function greedy_insert_circle_brute(
    circles::Vector{Circle{2, T}},
    new_radius::T,
    min_separation::T = zero(T),
) where {T}
    if isempty(circles)
        # Place first circle at origin
        return Circle(SA[zero(T), zero(T)], new_radius)
    end

    if length(circles) == 1
        # Place second circle at random angle, tangent to first circle
        ϕ = 2 * T(π) * rand(T)
        new_centre = centre(circles[1]) + (radius(circles[1]) + new_radius + min_separation) * SA[cos(ϕ), sin(ϕ)]
        return Circle(new_centre, new_radius)
    end

    # Brute force search for the best position
    best_energy = T(Inf)
    best_circle = Circle(SA[T(Inf), T(Inf)], new_radius)
    for i in 2:length(circles), j in 1:i-1
        # New candidate circles are tangent to ci and cj
        ci, cj = circles[i], circles[j]
        new_circles = tangent_circles(ci, cj, new_radius, min_separation)
        new_circles === nothing && continue

        # Check if the new circles are not overlapping with existing circles
        new_circle1, new_circle2 = new_circles
        if minimum(signed_overlap_distance(new_circle1, circles[k], min_separation) for k in 1:length(circles) if k != i && k != j; init = T(Inf)) >= 0
            packing_energy1 = packing_energy(circles, new_circle1)
            packing_energy1 < best_energy && ((best_energy, best_circle) = (packing_energy1, new_circle1))
        end
        if minimum(signed_overlap_distance(new_circle2, circles[k], min_separation) for k in 1:length(circles) if k != i && k != j; init = T(Inf)) >= 0
            packing_energy2 = packing_energy(circles, new_circle2)
            packing_energy2 < best_energy && ((best_energy, best_circle) = (packing_energy2, new_circle2))
        end
    end

    return best_circle
end

function greedy_circle_packing(radii::Vector{Float64}; min_separation::Float64 = 0.0)
    isempty(radii) && return Circle{2, Float64}[]

    # Greedily pack circles on at a time, placing the first circle at the origin
    circles = [Circle(SA[0.0, 0.0], radii[1])]
    for i in 2:length(radii)
        new_circle = greedy_insert_circle_brute(circles, radii[i], min_separation)
        push!(circles, new_circle)
    end

    # Translate circles such that the area-weighted mean is at the origin
    μ = area_weighted_mean(circles)
    for i in eachindex(circles)
        circles[i] = Circle(centre(circles[i]) - μ, radius(circles[i]))
    end

    return circles
end

####
#### Domain partitions
####

abstract type AbstractDomain{T} end

struct MyelinDomain{T} <: AbstractDomain{T}
    annulii::Vector{Annulus{2, T}}
    spheres::Vector{Circle{3, T}}
end

struct BloodVesselDomain{T} <: AbstractDomain{T}
    circles::Vector{Circle{2, T}}
end

@enum Region TissueRegion BloodRegion MyelinRegion AxonRegion FerritinRegion

regiondict() = Dict{String, Int}(string(r) => Int(r) for r in [AxonRegion, BloodRegion, MyelinRegion, TissueRegion, FerritinRegion])

findregion(x::T, y::T, domain::AbstractDomain{T}) where {T} = findregion(SA{T}[x, y], domain)

function findregion(x::SVector{2, T}, domain::BloodVesselDomain{T}) where {T}
    # Find the region that `x` is in
    (; circles) = domain
    i_inner = findfirst(c -> x ∈ c, circles)

    region = i_inner !== nothing ?
             BloodRegion : # in circle -> blood region
             TissueRegion # not in circle -> tissue region

    return region
end

function findregion(x::SVector{2, T}, domain::MyelinDomain{T}) where {T}
    # Find the region that `x` is in
    (; annulii, spheres) = domain

    # Find the region that `x` is in
    i_sphere = findfirst(c -> liftdim(x) ∈ c, spheres)
    i_outer = findfirst(c -> x ∈ outer_circle(c), annulii)
    i_inner = findfirst(c -> x ∈ inner_circle(c), annulii)

    region = if i_sphere !== nothing
        FerritinRegion # in sphere -> ferritin region (NOTE: FerritinRegion is contained within other regions)
    elseif i_outer === nothing
        @assert i_inner === nothing "Point is outside outer circles but inside an inner circle"
        TissueRegion # not in outer circles -> tissue region
    elseif i_inner !== nothing
        @assert i_outer == i_inner "Inner and outer circles are not corresponding"
        AxonRegion # in inner circles -> axon region
    else
        MyelinRegion # in outer circles but not inner circles -> myelin region
    end

    return region
end

####
#### Local frequency perturbation map functions
####

# Notation mapping from the horribly confusing notation in [1]:
#
#   - B field in x-direction (at θ = 0°) -> B field in z-direction (at θ = 0°)
#   - Cylinder in x-direction -> cylinder in z-direction
#   - y-z plane perpendicular to cylinder -> x-y plane perpendicular to cylinder
#   - Cartesian coordinates (z, y, x) -> (x, y, z) (not entirely sure about this...)
#   - ρ² = y² + z² -> ρ² = x² + y²
#   - θ = angle between cylinder and B-field -> same θ here
#   - θ3D = "azimuthal angle in the spherical coordinate system and r² ≡ x² + y² + z²", defined by cos(θ3D) = z / r when B⃗₀ = B₀ ẑ -> θ = angle between B⃗ and r⃗ (position relative to sphere origin), defined by cos(θ) = B̂ ⋅ r̂
#   - φ = 2D polar angle between y and z -> ϕ = 2D polar angle between x and y
#
# [1] Cheng Y-CN, Neelavalli J, Haacke EM. Limitations of Calculating Field Distributions and Magnetic Susceptibilities in MRI using a Fourier Based Method. Phys Med Biol 2009; 54: 1169–1189

struct OmegaDerivedConstants{T}
    ω₀::T
    s::T
    c::T
    s²::T
    c²::T
    function OmegaDerivedConstants(p::TissueParameters{T}) where {T}
        γ, B₀, θ = p.gamma, p.B0, p.theta
        ω₀ = γ * B₀
        s, c = sincos(θ)
        return new{T}(ω₀, s, c, s^2, c^2)
    end
end

function omega_blood_tissue(x::SVector{2, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c::Circle{2, T}) where {T}
    χv, a² = p.Chiv_Blood, radius(c)^2
    dx = x - centre(c)
    r² = norm2(dx)
    cos2ϕ = (dx[1] - dx[2]) * (dx[1] + dx[2]) / r² # cos2ϕ = (x² - y²) / r² = (x - y) * (x + y) / r²
    return b.ω₀ * χv * b.s² * (a² / r²) * cos2ϕ / 2
end

function omega_blood(x::SVector{2, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c::Circle{2, T}) where {T}
    χv = p.Chiv_Blood
    return b.ω₀ * χv * (3 * b.c² - 1) / 6 # constant offset within vessel
end

@inline function omega_myelin_tissue(x::SVector{2, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c_in::Circle{2, T}, c_out::Circle{2, T}) where {T}
    χI, χA, ri, ro = p.MyelinChiI, p.MyelinChiA, radius(c_in), radius(c_out)
    dx = x - centre(c_in)
    r² = norm2(dx)
    cos2ϕ = (dx[1] - dx[2]) * (dx[1] + dx[2]) / r² # cos2ϕ = (x² - y²) / r² = (x - y) * (x + y) / r²
    tmp = b.s² * cos2ϕ * ((ro - ri) * (ro + ri) / r²)
    I = χI / 2 * tmp # isotropic component
    A = χA / 8 * tmp # anisotropic component
    return b.ω₀ * (I + A)
end
@inline omega_myelin_tissue(x::SVector{2}, p::TissueParameters, b::OmegaDerivedConstants, a::Annulus{2}) = omega_myelin_tissue(x, p, b, inner_circle(a), outer_circle(a))

@inline function omega_myelin(x::SVector{2, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c_in::Circle{2, T}, c_out::Circle{2, T}) where {T}
    χI, χA, E, ri², ro = p.MyelinChiI, p.MyelinChiA, p.MyelinChiE, radius(c_in)^2, radius(c_out)
    dx = x - centre(c_in)
    r² = norm2(dx)
    r = √r²
    cos2ϕ = (dx[1] - dx[2]) * (dx[1] + dx[2]) / r² # cos2ϕ == (x²-y²)/r² == (x-y)*(x+y)/r²
    I = χI * (b.c² - T(1) / 3 - b.s² * cos2ϕ * (ri² / r²)) / 2 # isotropic component
    A = χA * (b.s² * (-T(5) / 12 - cos2ϕ / 8 * (1 + ri² / r²) + T(3) / 4 * log(ro / r)) - b.c² / 6) # anisotropic component
    return b.ω₀ * (I + A + E)
end
@inline omega_myelin(x::SVector{2}, p::TissueParameters, b::OmegaDerivedConstants, a::Annulus{2}) = omega_myelin(x, p, b, inner_circle(a), outer_circle(a))

@inline function omega_myelin_axon(x::SVector{2, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c_in::Circle{2, T}, c_out::Circle{2, T}) where {T}
    χA, ri, ro = p.MyelinChiA, radius(c_in), radius(c_out)
    A = 3 * χA / 4 * b.s² * log(ro / ri) # anisotropic (and only) component
    return b.ω₀ * A
end
@inline omega_myelin_axon(x::SVector{2}, p::TissueParameters, b::OmegaDerivedConstants, a::Annulus{2}) = omega_myelin_axon(x, p, b, inner_circle(a), outer_circle(a))

@inline function omega_ferritin_outside(x::SVector{3, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c::Circle{3, T}) where {T}
    χ, a = p.Chi_FeFull, radius(c)
    dx = x - centre(c)
    r = norm(dx)
    r̂ = dx / r
    B̂ = SA{T}[zero(T), b.s, b.c] # magnetic field direction
    cos²θ = (B̂ ⋅ r̂)^2 # polar angle with respect to B̂ is defined by cosθ = B̂ ⋅ r̂
    A = (χ / 3) * (3 * cos²θ - 1) * (a / r)^3 # field outside a sphere of constant susceptibility (Cheng Y-CN, Neelavalli J, Haacke EM. Limitations of Calculating Field Distributions and Magnetic Susceptibilities in MRI using a Fourier Based Method. Phys Med Biol 2009; 54: 1169–1189)
    return b.ω₀ * A
end

@inline function omega_ferritin_inside(x::SVector{3, T}, p::TissueParameters{T}, b::OmegaDerivedConstants{T}, c::Circle{3, T}) where {T}
    return zero(T) # no field inside a sphere of constant susceptibility
end

#### Analytic extremal values of the local frequency perturbation. Does not depend on size of respect structures (vessel/myelin/sphere inner/outer radii), but assumes only one structure is present.

function omega_blood_analytic_extremes(p::TissueParameters{T}) where {T}
    γ, B₀, θ = p.gamma, p.B0, p.theta
    χ = p.Chiv_Blood
    ω₀ = γ * B₀
    s, c = sincos(θ)
    ω_inside = ω₀ * χ * (3 * c^2 - 1) / 6 # constant within vessel `ω_inside`
    ω_outside = ω₀ * χ * s^2 / 2 # dipolar oscillation outside vessel bounded by `ω_outside`
    ω_extremes = (ω_inside, -ω_outside, ω_outside)
    return (; min = min(ω_extremes...), max = max(ω_extremes...))
end

function omega_myelin_analytic_extremes(p::TissueParameters{T}) where {T}
    γ, B₀, θ = p.gamma, p.B0, p.theta
    χI, χA, g = p.MyelinChiI, p.MyelinChiA, p.g_ratio
    ω₀ = γ * B₀
    s, c = sincos(θ)

    # For the annulus, we need to consider four permutations of possible extremal coordinates: r ∈ {r_inner, r_outer} and ϕ ∈ {0°, 90°}
    ωA_annulusᵢ₊ = ω₀ * χA * (s^2 * (T(-5 // 12) - T(+1 // 8) * 2 + T(3 // 4) * -log(g)) - c^2 / 6) # isotropic component with ϕ = 0° -> cos2ϕ = +1, r = r_inner
    ωA_annulusᵢ₋ = ω₀ * χA * (s^2 * (T(-5 // 12) - T(-1 // 8) * 2 + T(3 // 4) * -log(g)) - c^2 / 6) # isotropic component with ϕ = 90° -> cos2ϕ = -1, r = r_inner
    ωA_annulusₒ₊ = ω₀ * χA * (s^2 * (T(-5 // 12) - T(+1 // 8) * (1 + g^2)) - c^2 / 6) # isotropic component with ϕ = 0° -> cos2ϕ = +1, r = r_outer
    ωA_annulusₒ₋ = ω₀ * χA * (s^2 * (T(-5 // 12) - T(-1 // 8) * (1 + g^2)) - c^2 / 6) # isotropic component with ϕ = 90° -> cos2ϕ = -1, r = r_outer
    ωI_annulusᵢ₊ = ω₀ * χI * T(1 // 2) * (c^2 - T(1 // 3) - s^2) # isotropic component with ϕ = 0° -> cos2ϕ = +1, r = r_inner
    ωI_annulusᵢ₋ = ω₀ * χI * T(1 // 2) * (c^2 - T(1 // 3) + s^2) # isotropic component with ϕ = 90° -> cos2ϕ = -1, r = r_inner
    ωI_annulusₒ₊ = ω₀ * χI * T(1 // 2) * (c^2 - T(1 // 3) - s^2 * g^2) # isotropic component with ϕ = 0° -> cos2ϕ = +1, r = r_outer
    ωI_annulusₒ₋ = ω₀ * χI * T(1 // 2) * (c^2 - T(1 // 3) + s^2 * g^2) # isotropic component with ϕ = 90° -> cos2ϕ = -1, r = r_outer
    ω_annulusᵢ₊ = ωA_annulusᵢ₊ + ωI_annulusᵢ₊ # total field in annulus with ϕ = 0° -> cos2ϕ = +1, r = r_inner
    ω_annulusᵢ₋ = ωA_annulusᵢ₋ + ωI_annulusᵢ₋ # total field in annulus with ϕ = 90° -> cos2ϕ = -1, r = r_inner
    ω_annulusₒ₊ = ωA_annulusₒ₊ + ωI_annulusₒ₊ # total field in annulus with ϕ = 0° -> cos2ϕ = +1, r = r_outer
    ω_annulusₒ₋ = ωA_annulusₒ₋ + ωI_annulusₒ₋ # total field in annulus with ϕ = 90° -> cos2ϕ = -1, r = r_outer

    # The inner field is constant and the outer field is bounded by the dipolar oscillation
    ω_inside = ω₀ * χA * T(3 // 4) * s^2 * -log(g) # constant within myelin sheath (note: -log(g) > 0)
    ω_outside = ω₀ * (χI / 2 + χA / 8) * s^2 * (1 - g^2) # dipolar oscillation outside myelin sheath bounded by `ω_outside`

    ω_extremes = (ω_inside, -ω_outside, ω_outside, ω_annulusᵢ₊, ω_annulusᵢ₋, ω_annulusₒ₊, ω_annulusₒ₋)
    return (; min = min(ω_extremes...), max = max(ω_extremes...))
end

function omega_ferritin_analytic_extremes(p::TissueParameters{T}) where {T}
    γ, B₀ = p.gamma, p.B0
    χ = p.Chi_FeFull
    ω₀ = γ * B₀
    ω_inside = zero(T) # constant zero within sphere
    ω_outside₊ = ω₀ * χ * T(2 // 3) # maximum field occurs when position on sphere surface relative to sphere centre is parallel to B⃗
    ω_outside₋ = ω₀ * χ * T(-1 // 3) # minimum field occurs when position on sphere surface relative to sphere centre is perpendicular to B⃗
    ω_extremes = (ω_inside, ω_outside₋, ω_outside₊)
    return (; min = min(ω_extremes...), max = max(ω_extremes...))
end

####
#### Global frequency perturbation functions: calculate ω(x) due to entire domain
####

# Calculate ω(x) by searching for the region which `x` is contained in
function omegamap(
    x::SVector{2, T},
    p::TissueParameters{T},
    domain::BloodVesselDomain{T},
) where {T}

    # If there are no structures, then there is no frequency shift ω
    (; circles) = domain
    isempty(circles) && return zero(eltype(x))

    constants = OmegaDerivedConstants(p)
    ω = zero(T)

    # Find the region that `x` is in
    i_inner = findfirst(c -> x ∈ c, circles)

    region = if i_inner !== nothing
        BloodRegion # in circle -> blood region
    else
        TissueRegion # not in circle -> tissue region
    end

    # Add contributions from myelin sheaths
    if region == TissueRegion
        @inbounds for i in eachindex(circles)
            ω += omega_blood_tissue(x, p, constants, circles[i])
        end
    else # region == BloodRegion
        @inbounds for i in eachindex(circles)
            ω += i == i_inner ?
                 omega_blood(x, p, constants, circles[i]) :
                 omega_blood_tissue(x, p, constants, circles[i])
        end
    end

    return ω
end

# Calculate ω(x) by searching for the region which `x` is contained in
function omegamap(
    x::SVector{2, T},
    p::TissueParameters{T},
    domain::MyelinDomain{T},
) where {T}

    # If there are no structures, then there is no frequency shift ω
    (; annulii, spheres) = domain
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
        x3D = liftdim(x) # pad to 3D
        ω += x3D ∈ spheres[i] ?
             omega_ferritin_inside(x3D, p, constants, spheres[i]) :
             omega_ferritin_outside(x3D, p, constants, spheres[i])
    end

    return ω
end

omegamap(x::T, y::T, p::TissueParameters{T}, domain::AbstractDomain{T}) where {T} = omegamap(SA{T}[x, y], p, domain)

function omegamap(x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::AbstractDomain{T}) where {T}
    ω = zeros(T, length(x), length(y))
    Threads.@threads for j in eachindex(y)
        for i in eachindex(x)
            ω[i, j] = omegamap(x[i], y[j], p, domain)
        end
    end
    return ω
end

####
#### Map over domain by region
####

function mapdomain(f, x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::AbstractDomain{T}) where {T}
    out = zeros(T, length(x), length(y))
    Threads.@threads for j in eachindex(y)
        for i in eachindex(x)
            x⃗ = SA{T}[x[i], y[j]]
            region = findregion(x⃗, domain)
            out[i, j] = f(x⃗, p, region)
        end
    end
    return out
end

function regionmap(x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::AbstractDomain{T}) where {T}
    return mapdomain(x, y, p, domain) do _, _, region
        return Int(region)
    end
end

function t2map(x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::BloodVesselDomain{T}) where {T}
    return mapdomain(x, y, p, domain) do _, p, region
        if region == BloodRegion
            inv(p.R2v_Blood)
        else # region == TissueRegion
            inv(p.R2_Tissue)
        end
    end
end

function t1map(x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::BloodVesselDomain{T}) where {T}
    return mapdomain(x, y, p, domain) do _, p, region
        if region == BloodRegion
            inv(p.R1_Blood)
        else # region == TissueRegion
            inv(p.R1_Tissue)
        end
    end
end

function t2map(x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::MyelinDomain{T}) where {T}
    return mapdomain(x, y, p, domain) do _, p, region
        if region == FerritinRegion
            inv(p.R2_Fe)
        elseif region == TissueRegion
            inv(p.R2_lp)
        elseif region == AxonRegion
            inv(p.R2_lp)
        else # region == MyelinRegion
            inv(p.R2_sp)
        end
    end
end

function t1map(x::AbstractVector{T}, y::AbstractVector{T}, p::TissueParameters{T}, domain::MyelinDomain{T}) where {T}
    return mapdomain(x, y, p, domain) do _, p, region
        if region == FerritinRegion
            inv(p.R1_Fe)
        elseif region == TissueRegion
            inv(p.R1_lp)
        elseif region == AxonRegion
            inv(p.R1_lp)
        else # region == MyelinRegion
            inv(p.R1_sp)
        end
    end
end

end # module SimulatedFields
