module HyperparametersAbstractGPs

using ParameterHandling
using Optim
using Zygote
using AbstractGPs

export BoundedHyperparameters, optimize_hyperparameters

const default_optimizer = LBFGS(;
    alphaguess = Optim.LineSearches.InitialStatic(; scaled = true),
    linesearch = Optim.LineSearches.BackTracking())

"""
    BoundedHyperparameters(compute_initial_points::Function;
        optimizer = default_optimizer,
        maxiter = 1_000)

    BoundedHyperparameters(const_inital_point::NamedTuple;
        optimizer = default_optimizer,
        maxiter = 1_000)

Hyperparameters defined on bounded intervals.

Stores a function `compute_initial_points` and options for an optimization algorithm
from package `Optim`.
For `xs` points in the domain and `ys` function values at `xs`, `compute_initial_points(xs,ys)`
returns a vector of `NamedTuple` elements, where names are equal to hyperparameters
and corresponding values are of type `ParameterHandling.Bounded`.
Elements of such a returned vector are used as initial points for hyperparameter optimizer,
hence initial points can be computed from function evaluation history.

If you wish to always start the optimizer with a single constant initial point, please see the
example below.

# Examples:

```julia-repl
julia> using ParameterHandling
julia> BoundedHyperparameters((;
        lengthscales = bounded(ones(4), 0.004, 4),
        signal_var = bounded(1.0, 0.01, 15.0),
        noise_var = bounded(0.1, 0.0001, 0.2)))
```
"""
struct BoundedHyperparameters{T}
    compute_initial_points::Function
    # Optim parameters
    optimizer::T
    maxiter::Int
end

function BoundedHyperparameters(compute_initial_points::Function;
    optimizer = default_optimizer,
    maxiter = 1_000)
    BoundedHyperparameters(compute_initial_points, optimizer, maxiter)
end

function BoundedHyperparameters(const_inital_point::NamedTuple;
    optimizer = default_optimizer,
    maxiter = 1_000)
    BoundedHyperparameters((xs, ys) -> [const_inital_point], optimizer, maxiter)
end

# code adopted from:
# https://juliagaussianprocesses.github.io/AbstractGPs.jl/dev/examples/1-mauna-loa/
# and https://github.com/JuliaGaussianProcesses/ParameterHandling.jl
"""
    optimize_hyperparameters(xs,
        ys,
        kernel_creator,
        unif_prior::BoundedHyperparameters)

Optimize hyperparameters using a prior of type `BoundedHyperparameters`.

See also [`BoundedHyperparameters`](@ref).
"""
function optimize_hyperparameters(xs,
    ys,
    kernel_creator,
    unif_prior::BoundedHyperparameters)
    loss = setup_loss(xs, ys, kernel_creator)
    initial_points = unif_prior.compute_initial_points(xs, ys)
    current_minimum = Inf
    current_minimizer = nothing
    for θ_initial in initial_points
        proposed_minimizer, proposed_minimum = minimize(loss, θ_initial)
        if proposed_minimum < current_minimum
            current_minimum = proposed_minimum
            current_minimizer = proposed_minimizer
        end
    end
    return ParameterHandling.value(current_minimizer)
end

setup_loss(xs, ys, kernel_creator) = θ -> negative_lml(xs, ys, kernel_creator, θ)

function negative_lml(xs, ys, kernel_creator, θ)
    # prior process
    f = GP(kernel_creator(ParameterHandling.value(θ)))
    # finite projection at xs
    if :noise_var in keys(θ)
        fx = f(xs, ParameterHandling.value(θ.noise_var))
    else
        fx = f(xs)
    end
    # negative log marginal likelihood of posterior
    -logpdf(fx, ys)
end

function minimize(loss, θ_initial; optimizer = default_optimizer, maxiter = 1_000)
    options = Optim.Options(; iterations = maxiter, show_trace = false)

    θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_initial)
    # unflatten: vector -> NamedTuple
    # loss: NamedTuple -> Float64
    loss_packed = loss ∘ unflatten

    # https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations
    function fg!(F, G, x)
        if F !== nothing && G !== nothing
            val, grad = Zygote.withgradient(loss_packed, x)
            G .= only(grad)
            return val
        elseif G !== nothing
            grad = Zygote.gradient(loss_packed, x)
            G .= only(grad)
            return nothing
        elseif F !== nothing
            return loss_packed(x)
        end
    end
    # TODO: what if it fails?? does it throw an error?
    result = optimize(Optim.only_fg!(fg!), θ_flat_init, optimizer, options; inplace = false)
    return unflatten(result.minimizer), Optim.minimum(result)
end

end # module HyperparametersAbstractGPs
