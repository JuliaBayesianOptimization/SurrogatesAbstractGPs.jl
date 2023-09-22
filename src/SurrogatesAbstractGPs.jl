module SurrogatesAbstractGPs
# Refactoring and extending registered SurrogatesAbstractGPs package,
# https://github.com/SciML/Surrogates.jl/tree/e6aa022e612ac57228506e625c662438d385e69d/lib/SurrogatesAbstractGPs

# currently SurrogatesBase is from a fork https://github.com/samuelbelko/SurrogatesBase.jl.git
# on branch at_point
using SurrogatesBase
# adding methods to these functions
import SurrogatesBase: add_point!, add_points!,
    update_hyperparameters!, hyperparameters,
    mean, mean_at_point,
    var, var_at_point,
    mean_and_var, mean_and_var_at_point,
    rand, rand_at_point

import AbstractGPs
import KernelFunctions

include("HyperparametersAbstractGPs.jl")
using .HyperparametersAbstractGPs

export BoundedHyperparameters
export GPSurrogate
# SurrogatesBase interface
export add_point!, add_points!,
    update_hyperparameters!, hyperparameters,
    mean, mean_at_point,
    var, var_at_point,
    mean_and_var, mean_and_var_at_point,
    rand, rand_at_point

include("utils.jl")

# GPSurrogate for functions defined on R^n, D has to be constrained, as otherwise it could
# be Any and so the definition of mean would become ambiguous with the one in Statistics
mutable struct GPSurrogate{D <: Union{Number, AbstractVector}, R, GP, GP_P, F} <:
               AbstractSurrogate
    xs::Vector{D}
    ys::Vector{R}
    # prior process
    gp::GP
    gp_posterior::GP_P
    hyperparameters::NamedTuple
    kernel_creator::F
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
    return GPSurrogate(xs,
        ys,
        gp,
        AbstractGPs.posterior(gp(xs, hyperparameters.noise_var), ys),
        hyperparameters,
        kernel_creator)
end

# for add_point! copies of xs and ys need to be made because we get
# "Error: cannot resize array with shared data " if we push! directly to xs and ys
function add_point!(g::GPSurrogate, new_x, new_y)
    x_copy = copy(g.xs)
    push!(x_copy, new_x)
    y_copy = copy(g.ys)
    push!(y_copy, new_y)
    updated_posterior = AbstractGPs.posterior(g.gp(x_copy, g.hyperparameters.noise_var),
        y_copy)
    g.xs, g.ys, g.gp_posterior = x_copy, y_copy, updated_posterior
    return nothing
end

function add_points!(g::GPSurrogate, new_xs::Vector, new_ys::Vector)
    length(new_xs) == length(new_ys) ||
        throw(ArgumentError("new_xs, new_ys have different lengths"))
    x_copy = copy(g.xs)
    append!(x_copy, new_xs)
    y_copy = copy(g.ys)
    append!(y_copy, new_ys)
    updated_posterior = AbstractGPs.posterior(g.gp(x_copy, g.hyperparameters.noise_var),
        y_copy)
    g.xs, g.ys, g.gp_posterior = x_copy, y_copy, updated_posterior
    return nothing
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
    return nothing
end

hyperparameters(g::GPSurrogate) = g.hyperparameters

function mean_at_point(g::GPSurrogate, x)
    return only(AbstractGPs.mean(g.gp_posterior([x])))
end
# mean at points
function mean(g::GPSurrogate, xs::Vector)
    return AbstractGPs.mean(g.gp_posterior(xs))
end

# variance at point
function var_at_point(g::GPSurrogate, x)
    return only(AbstractGPs.var(g.gp_posterior([x])))
end
# variance at points
function var(g::GPSurrogate, xs::Vector)
    return AbstractGPs.var(g.gp_posterior(xs))
end

# mean and variance at point
function mean_and_var_at_point(g::GPSurrogate, x)
    return only.(AbstractGPs.mean_and_var(g.gp_posterior([x])))
end
# mean and variance at points
function mean_and_var(g::GPSurrogate, xs::Vector)
    return AbstractGPs.mean_and_var(g.gp_posterior(xs))
end

# sample from joint posterior, use default in SurrogatesBase for "at point" version
function rand(g::GPSurrogate, xs::Vector)
    return rand(g.gp_posterior(xs))
end

end # module
