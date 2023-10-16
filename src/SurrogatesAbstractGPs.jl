module SurrogatesAbstractGPs
# Refactoring and extending registered SurrogatesAbstractGPs package,
# https://github.com/SciML/Surrogates.jl/tree/e6aa022e612ac57228506e625c662438d385e69d/lib/SurrogatesAbstractGPs

# currently SurrogatesBase is from a fork https://github.com/samuelbelko/SurrogatesBase.jl.git#param-abstract-type
# (on branch param-abstract-type)
using SurrogatesBase
import SurrogatesBase: add_points!,
    update_hyperparameters!, hyperparameters,
    mean, var, mean_and_var, rand

import AbstractGPs
import KernelFunctions

include("HyperparametersAbstractGPs.jl")
using .HyperparametersAbstractGPs

export BoundedHyperparameters
export GPSurrogate, add_points!
export update_hyperparameters!, hyperparameters
export mean, var, mean_and_var, rand

include("utils.jl")

# GPSurrogate for functions defined on R^n, D has to be constrained, as otherwise it could
# be Any and so the definition of mean would become ambiguous with the one in Statistics
mutable struct GPSurrogate{D, R, GP, GP_P, H, F} <: AbstractSurrogate
    xs::Vector{D}
    ys::Vector{R}
    gp::GP
    gp_posterior::GP_P
    hyperparameters::H
    kernel_creator::F
end

function Base.show(io::IO, ::MIME"text/plain", g::GPSurrogate)
    println(io, "GPSurrogate on domain $(eltype(g.xs)) and range $(eltype(g.ys))")
    println(io, "GP: $(typeof(g.gp))")
    println(io, "Hyperparameters: $(g.hyperparameters)")
    println(io, "Number of observations: $(length(g.xs))")
end

"""
    GPSurrogate(xs,
        ys;
        kernel_creator = (_ -> KernelFunctions.Matern52Kernel()),
        hyperparameters = (; noise_var = 0.1))

Gaussian process surrogate.

Pass points `xs` with corresponding evaluations in `ys`,
`kernel_creator` function and `hyperparameters` of type `NamedTuple`.

`kernel_creator` needs to map `hyperparameters` into a kernel function as defined by the
package KernelFunctions.jl.

If `hyperparameters` includes an entry with name `noise_var`, then the value of `noise_var`
will be passed directly to `AbstractGPs`, hence the `kernel_creator` should never use
the entry `noise_var` inside `hyperparameters`. Please compare with [Mauna loa example](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/examples/1-mauna-loa/#Posterior)
in `AbstractGPs` docs.
"""
function GPSurrogate(xs,
    ys;
    kernel_creator = (_ -> KernelFunctions.Matern52Kernel()),
    hyperparameters = (; noise_var = 0.1))
    # prior process, for safety remove noise_var from hyperparameters when passing to
    # kernel_creator, see docs above
    gp = AbstractGPs.GP(kernel_creator(delete(hyperparameters, :noise_var)))
    # if :noise_var is not in keys(hyperparameters), add entry noise_var = 0.0
    hyperparameters = merge((; noise_var = 0.0), hyperparameters)
    GPSurrogate(copy(xs),
                copy(ys),
                gp,
                AbstractGPs.posterior(gp(copy(xs), hyperparameters.noise_var), copy(ys)),
                hyperparameters,
                kernel_creator)
end

function add_points!(g::GPSurrogate, new_xs, new_ys)
    length(new_xs) == length(new_ys) ||
        throw(ArgumentError("new_xs, new_ys have different lengths"))
    append!(g.xs, new_xs)
    append!(g.ys, new_ys)
    g.gp_posterior = AbstractGPs.posterior(AbstractGPs.FiniteGP(g.gp_posterior,
                                                    new_xs,
                                                    g.hyperparameters.noise_var),
                                           new_ys)
    g
end

"""
    update_hyperparameters!(g::GPSurrogate, prior)

Hyperparameter tuning for `GPSurrogate`.

See also [`BoundedHyperparameters`](@ref) that can be used as a prior.
"""
function update_hyperparameters!(g::GPSurrogate, prior)
    # save new hyperparameters
    g.hyperparameters = optimize_hyperparameters(g.xs, g.ys, g.kernel_creator, prior)
    # update GP and its posterior
    g.gp = AbstractGPs.GP(g.kernel_creator(g.hyperparameters))
    g.gp_posterior = AbstractGPs.posterior(g.gp(g.xs, g.hyperparameters.noise_var), g.ys)
    g
end

hyperparameters(g::GPSurrogate) = g.hyperparameters

# mean at point, have to add <: Number, otherwise there is ambiguity with mean from Statistics
mean(g::GPSurrogate, xs::AbstractVector) = AbstractGPs.mean(g.gp_posterior(xs))

# variance at point
var(g::GPSurrogate, xs::AbstractVector) = AbstractGPs.var(g.gp_posterior(xs))

# mean and variance at point
mean_and_var(g::GPSurrogate, xs::AbstractVector) = AbstractGPs.mean_and_var(g.gp_posterior(xs))
#
# sample from joint posterior, use default in SurrogatesBase for "at point" version
rand(g::GPSurrogate, xs::AbstractVector) = rand(g.gp_posterior(xs))

end # module
