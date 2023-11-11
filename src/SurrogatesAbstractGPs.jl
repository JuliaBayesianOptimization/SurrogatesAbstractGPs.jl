module SurrogatesAbstractGPs
# Refactoring and extending registered SurrogatesAbstractGPs package,
# https://github.com/SciML/Surrogates.jl/tree/e6aa022e612ac57228506e625c662438d385e69d/lib/SurrogatesAbstractGPs

# currently SurrogatesBase is from a fork https://github.com/samuelbelko/SurrogatesBase.jl.git#finite_posterior
# (on branch finite_posterior)

using SurrogatesBase
import AbstractGPs
import KernelFunctions

include("HyperparametersAbstractGPs.jl")
using .HyperparametersAbstractGPs

export BoundedHyperparameters
export GPSurrogate
export add_points!, update_hyperparameters!, hyperparameters, finite_posterior

# reexport from AbstractGPs
import AbstractGPs: mean, var, mean_and_var, rand
export mean, var, mean_and_var, rand

include("utils.jl")

mutable struct GPSurrogate{D, R, GP_P, H <: NamedTuple, F} <:
               AbstractStochasticSurrogate
    xs::Vector{D}
    ys::Vector{R}
    gp_posterior::GP_P
    hyperparameters::H
    kernel_creator::F
end

function Base.show(io::IO, ::MIME"text/plain", g::GPSurrogate)
    println(io, "GPSurrogate on domain $(eltype(g.xs)) and range $(eltype(g.ys)).")
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

The constructor accepts a nonempty vector `xs` with corresponding function evaluations in `ys`
of the same length, `kernel_creator` function and `hyperparameters` of type `NamedTuple`.

`kernel_creator` needs to map `hyperparameters` into a kernel function from the
package `KernelFunctions.jl`.

If `hyperparameters` includes an entry with key `noise_var`, then the value of `noise_var`
will be passed directly to `AbstractGPs`, hence the `kernel_creator` should never use
the entry `noise_var` inside `hyperparameters`. Please compare with [Mauna loa example](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/examples/1-mauna-loa/#Posterior)
in `AbstractGPs` docs.
"""
function GPSurrogate(xs,
    ys;
    kernel_creator = (_ -> KernelFunctions.Matern52Kernel()),
    hyperparameters = (; noise_var = 0.1))
    length(xs) == length(ys) ||
        throw(ArgumentError("xs, ys have different lengths"))
    length(xs) == 0 &&
        throw(ArgumentError("xs and ys are empty"))

    # if :noise_var is not in keys(hyperparameters), add entry noise_var = 1e-18
    if !(:noise_var in keys(hyperparameters))
        hyperparameters = merge(hyperparameters, (; noise_var = 1e-18))
    end

    # prior process, for safety remove noise_var from hyperparameters when passing to
    # kernel_creator, see docs of GPSurrogate constructor
    prior = AbstractGPs.GP(kernel_creator(delete(hyperparameters, :noise_var)))
    posterior = AbstractGPs.posterior(prior(copy(xs), hyperparameters.noise_var), copy(ys))

    return GPSurrogate(copy(xs),
        copy(ys),
        posterior,
        hyperparameters,
        kernel_creator)
end

function SurrogatesBase.add_points!(g::GPSurrogate, new_xs, new_ys)
    length(new_xs) == length(new_ys) ||
        throw(ArgumentError("new_xs, new_ys have different lengths"))
    length(new_xs) == 0 &&
        throw(ArgumentError("new_xs and new_ys are empty"))

    append!(g.xs, new_xs)
    append!(g.ys, new_ys)
    # efficient sequential conditioning, see https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/concrete_features/#Sequential-Conditioning
    g.gp_posterior = AbstractGPs.posterior(g.gp_posterior(new_xs,
            g.hyperparameters.noise_var),
        new_ys)
    return g
end

"""
    update_hyperparameters!(g::GPSurrogate, prior)

Maximize the log marginal likelihood with respect to the hyperparameters.

See also [`BoundedHyperparameters`](@ref) that can be used as a prior.
"""
function SurrogatesBase.update_hyperparameters!(g::GPSurrogate, prior)
    # save new hyperparameters
    g.hyperparameters = optimize_hyperparameters(g.xs, g.ys, g.kernel_creator, prior)
    # prior process, for safety remove noise_var from hyperparameters when passing to
    # kernel_creator, see docs of GPSurrogate constructor
    prior = AbstractGPs.GP(g.kernel_creator(delete(g.hyperparameters, :noise_var)))
    # update posterior
    g.gp_posterior = AbstractGPs.posterior(prior(copy(g.xs), g.hyperparameters.noise_var),
        copy(g.ys))
    return nothing
end

SurrogatesBase.hyperparameters(g::GPSurrogate) = g.hyperparameters

"""
    finite_posterior(g::GPSurrogate, xs)

Returned object supports:

- `mean(finite_posterior(g, xs))` returns a vector of posterior means at `xs`
- `var(finite_posterior(g, xs))` returns a vector of posterior variances at `xs`
- `mean_and_var(finite_posterior(g, xs))` returns a `Tuple` consisting of a vector
of posterior means and a vector of posterior variances at `xs`
- `rand(finite_posterior(g, xs))` returns a sample from the joint posterior at points `xs`
"""
SurrogatesBase.finite_posterior(g::GPSurrogate, xs) = (g.gp_posterior)(xs, g.hyperparameters.noise_var)

end # module
