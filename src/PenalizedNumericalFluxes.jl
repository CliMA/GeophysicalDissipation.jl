module PenalizedNumericalFluxes

using LinearAlgebra
using StaticArrays

using ClimateMachine.VariableTemplates
using ClimateMachine.DGMethods.NumericalFluxes: NumericalFluxFirstOrder
using ClimateMachine.BalanceLaws

import ClimateMachine.DGMethods.NumericalFluxes: numerical_flux_first_order!

@inline first_order_flux!(F, balance_law, U, A, time, direction) =
    flux_first_order!(balance_law, F, U, A, time, direction)

abstract type AbstractPenalizedNumericalFlux <: NumericalFluxFirstOrder end

function numerical_flux_first_order!(numerical_flux::AbstractPenalizedNumericalFlux,
                                     balance_law::BalanceLaw,
                                     F̂ᵀn::Vars{S},
                                     n⁻::SVector,
                                     U⁻::Vars{S},
                                     A⁻::Vars{A},
                                     U⁺::Vars{S},
                                     A⁺::Vars{A},
                                     time,
                                     direction) where {S, A}
   
    F̂ᵀn = parent(F̂ᵀn)

    N = number_states(balance_law, Prognostic())
    F⁻ = similar(n_dot_F̂ᵀ, Size(3, N))
    F⁺ = similar(n_dot_F̂ᵀ, Size(3, N))

    FT = eltype(n_dot_F̂ᵀ)
    fill!(F⁻, -zero(FT))
    fill!(F⁺, -zero(FT))

    first_order_flux!(Grad{S}(F⁻), balance_law, U⁻, A⁻, time, direction)
    first_order_flux!(Grad{S}(F⁺), balance_law, U⁺, A⁺, time, direction)

    c = penalty_matrix(numerical_flux, balance_law, n⁻, U⁻, A⁻, U⁺, A⁺, time, direction)

    F̂ᵀn .+= (F⁻ + F⁺)' * n⁻ / 2 + c * (parent(U⁻) - parent(U⁺))
                    
    return nothing
end

#####
##### Component-constant, direction-independent penalty
#####

struct ConstantPenalization{FT} <: AbstractPenalizedNumericalFlux
    speed :: FT
end

@inline function penalty_matrix(penalty::ConstantPenalization,
                                balance_law, n⁻, U⁻, A⁻, U⁺, A⁺, time, direction)
    FT = typeof(time)
    N = number_states(balance_law, Prognostic())

    return SDiagonal(ntuple(i -> penalty.speed, Val(N)))
end

#####
##### Component-constant, direction-dependent penalty
#####

struct ConstantDirectionalPenalization{FT} <: AbstractPenalizedNumericalFlux
    cˣ :: FT
    cʸ :: FT
    cᶻ :: FT
end

@inline function penalty_matrix(penalty::ConstantDirectionalPenalization,
                                balance_law, n⁻, U⁻, A⁻, U⁺, A⁺, time, direction)
    FT = typeof(time)
    N = number_states(balance_law, Prognostic())

    cˣ = penalty.cˣ
    cʸ = penalty.cʸ
    cᶻ = penalty.cᶻ

    return SDiagonal(ntuple(i -> abs(SVector(cˣ, cʸ, cᶻ)' * n⁻), Val(N)))
end

end # module
