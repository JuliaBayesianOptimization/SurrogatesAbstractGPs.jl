module SurrogatesAbstractGPs
# Refactoring and extending SurrogatesAbstractGPs package,
# https://github.com/SciML/Surrogates.jl/tree/e6aa022e612ac57228506e625c662438d385e69d/lib/SurrogatesAbstractGPs

# currently SurrogatesBase is from a fork https://github.com/samuelbelko/SurrogatesBase.jl.git
import SurrogatesBase:
    AbstractSurrogate,
    add_point!,
    add_points!,
    supports_hyperparameters,
    update_hyperparameters!,
    hyperparameters,
    supports_posterior,
    posterior
import AbstractGPs
import KernelFunctions

export GPSurrogate,
    add_point!,
    add_points!,
    supports_hyperparameters,
    update_hyperparameters!,
    hyperparameters,
    supports_posterior,
    posterior,
    mean_at_point,
    std_error_at_point,
    logpdf_surrogate

include("HyperparametersAbstractGPs.jl")
using .HyperparametersAbstractGPs
export BoundedHyperparameters

include("utils.jl")

mutable struct GPSurrogate{X, Y, GP, GP_P, F} <: AbstractSurrogate
    x::X
    y::Y
    # prior process
    gp::GP
    gp_posterior::GP_P
    hyperparameters::NamedTuple
    kernel_creator::F
end
"""
    GPSurrogate(x,
        y;
        kernel_creator = (_ -> KernelFunctions.Matern52Kernel()),
        hyperparameters = (; noise_var = 0.1))

Gaussian process surrogate.

Pass points `x` with corresponding evaluations in `y`,
`kernel_creator` function and `hyperparameters` of type `NamedTuple`.

`kernel_creator` needs to map `hyperparameters` into a kernel function as defined by the
package KernelFunctions.jl.

If `hyperparameters` includes an entry with name `noise_var`, then the value of `noise_var`
will be passed directly to `AbstractGPs`, hence the `kernel_creator` should never use
the entry `noise_var` inside `hyperparameters`. Please compare with [Mauna loa example](https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/examples/1-mauna-loa/#Posterior)
in `AbstractGPs` docs.
"""
function GPSurrogate(x,
    y;
    kernel_creator = (_ -> KernelFunctions.Matern52Kernel()),
    hyperparameters = (; noise_var = 0.1))
    # prior process, for safety remove noise_var from hyperparameters, see docs above
    gp = AbstractGPs.GP(kernel_creator(delete(hyperparameters, :noise_var)))
    # TODO: document:  noise variance passed into AbstractGPs and not into kernel_creator,
    # cf. https://juliagaussianprocesses.github.io/AbstractGPs.jl/stable/examples/1-mauna-loa/#Posterior
    # if :noise_var is not in keys(hyperparameters), add entry noise_var = 0.0
    hyperparameters = merge((; noise_var = 0.0), hyperparameters)
    GPSurrogate(x,
        y,
        gp,
        AbstractGPs.posterior(gp(x, hyperparameters.noise_var), y),
        hyperparameters,
        kernel_creator)
end

# for add_point! copies of x and y need to be made because we get
# "Error: cannot resize array with shared data " if we push! directly to x and y
function add_point!(g::GPSurrogate, new_x, new_y)
    x_copy = copy(g.x)
    push!(x_copy, new_x)
    y_copy = copy(g.y)
    push!(y_copy, new_y)
    updated_posterior = AbstractGPs.posterior(g.gp(x_copy, g.hyperparameters.noise_var),
        y_copy)
    g.x, g.y, g.gp_posterior = x_copy, y_copy, updated_posterior
    nothing
end

function add_points!(g::GPSurrogate, new_xs, new_ys)
    length(new_xs) == length(new_ys) ||
        throw(ArgumentError("new_xs, new_ys have different lengths"))
    x_copy = copy(g.x)
    append!(x_copy, new_xs)
    y_copy = copy(g.y)
    append!(y_copy, new_ys)
    updated_posterior = AbstractGPs.posterior(g.gp(x_copy, g.hyperparameters.noise_var),
        y_copy)
    g.x, g.y, g.gp_posterior = x_copy, y_copy, updated_posterior
    nothing
end

function mean_at_point(g::GPSurrogate, val)
    # Check to make sure dimensions of input matches expected dimension of surrogate
    _check_dimension(g, val)
    return only(AbstractGPs.mean(g.gp_posterior([val])))
end

function std_error_at_point(g::GPSurrogate, val)
    _check_dimension(g, val)
    return sqrt(only(AbstractGPs.var(g.gp_posterior([val]))))
end

"""
    logpdf_surrogate(g::GPSurrogate)

Log marginal posterior predictive probability.
"""
function logpdf_surrogate(g::GPSurrogate)
    # Here was a bug,
    # logpdf(g.gp_posterior(g.x), g.y) returns logpdf of g.gp_posterior(g.x) (as a prior),
    # conditioned again on g.y observations at g.x
    return AbstractGPs.logpdf(g.gp(g.x, g.hyperparameters.noise_var), g.y)
end

supports_posterior(g::GPSurrogate) = true
"""
    posterior(s::GPSurrogate, x)

Return a joint posterior at `x = [x_1, ..., x_m]`, sample via `rand(posterior(s,x))`
for `x = [x_1,...,x_m]`.
"""
posterior(g::GPSurrogate, x) = g.gp_posterior(x)

supports_hyperparameters(g::GPSurrogate) = true
hyperparameters(g::GPSurrogate) = g.hyperparameters
"""
    update_hyperparameters!(g::GPSurrogate, prior)

Hyperparameter tuning for `GPSurrogate`.

See also [`BoundedHyperparameters`](@ref) that can be used as a prior.
"""
function update_hyperparameters!(g::GPSurrogate, prior)
    # save new hyperparameters
    g.hyperparameters = optimize_hyperparameters(g.x, g.y, g.kernel_creator, prior)
    # update GP and its posterior
    g.gp = AbstractGPs.GP(g.kernel_creator(g.hyperparameters))
    g.gp_posterior = AbstractGPs.posterior(g.gp(g.x, g.hyperparameters.noise_var), g.y)
    nothing
end

end # module
