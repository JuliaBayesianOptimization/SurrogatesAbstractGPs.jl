using SurrogatesAbstractGPs
using Test
using ParameterHandling
using KernelFunctions
import AbstractGPs

include("../src/utils.jl")

unif_prior = BoundedHyperparameters([(;
    lengthscale = bounded(1.0, 0.004, 4.0),
    noise_var = bounded(0.1, 0.0001, 0.2))])

function kernel_creator(hyperparameters)
    return with_lengthscale(KernelFunctions.Matern52Kernel(), hyperparameters.lengthscale)
end

# 1-dim surrogate
s = GPSurrogate([1.0, 3.0, 4.0], [0.9, 3.1, 4.5],
    kernel_creator = kernel_creator,
    hyperparameters = (; lengthscale = 1.0, noise_var = 0.1))

@testset "add_points!, 1-dim" begin
    add_points!(s, [5.0], [5.9])
    add_points!(s, [6.0, 7.1], [5.9, 6.9])
    @test length(s.xs) == 6
    @test all(s.xs .== [1.0, 3.0, 4.0, 5.0, 6.0, 7.1])
    @test all(s.ys .== [0.9, 3.1, 4.5, 5.9, 5.9, 6.9])
end

@testset "mean at point, 1-dim" begin
    @test 1 < only(mean(finite_posterior(s, [2.0]))) < 3
end

@testset "mean at points, 1-dim" begin
    @test length(mean(finite_posterior(s, [2.0, 4.3]))) == 2
end

@testset "var at point, 1-dim" begin
    @test 0 <= only(var(finite_posterior(s, [1.0]))) <= 1
end

@testset "var at points, 1-dim" begin
    @test length(var(finite_posterior(s, [2.0, 4.3, 0.4]))) == 3
end

@testset "mean_and_var at point, 1-dim" begin
    μ, σ² = only.(mean_and_var(finite_posterior(s, [1.0])))
    @test 0.7 < μ < 1.1
    @test 0 <= σ² <= 1
end

@testset "mean_and_var at points, 1-dim" begin
    μs, σ²s = mean_and_var(finite_posterior(s, [1.0, 3.0]))
    @test length(μs) == 2
    @test length(σ²s) == 2
end

@testset "read hyperparameters, 1-dim" begin
    @test hyperparameters(s).lengthscale == 1.0
    @test hyperparameters(s).noise_var == 0.1
    @test length(keys(hyperparameters(s))) == 2
end

@testset "update_hyperparameters!, 1-dim" begin
    old_prior = AbstractGPs.GP(s.kernel_creator(delete(s.hyperparameters, :noise_var)))
    old_logpdf = AbstractGPs.logpdf(old_prior(s.xs, hyperparameters(s).noise_var), s.ys)
    update_hyperparameters!(s, unif_prior)
    new_prior = AbstractGPs.GP(s.kernel_creator(delete(s.hyperparameters, :noise_var)))
    new_logpdf = AbstractGPs.logpdf(new_prior(s.xs, hyperparameters(s).noise_var), s.ys)
    # new hyperparameters should make data more probable, maximize marginal logpdf
    @test old_logpdf <= new_logpdf
end

@testset "rand at point, 1-dim" begin
    @test isa(only(rand(finite_posterior(s, [1.3]))), Number)
end

@testset "rand at points, 1-dim" begin
    @test length(rand(finite_posterior(s, [1.3, 4.0]))) == 2
end

# 2-dim surrogate
d = GPSurrogate([[5.0, 4.0]], [5])
unif_prior = BoundedHyperparameters([(noise_var = bounded(0.1, 0.0001, 0.2),)])
add_points!(d, [[6.0, 7.1], [5.0, 21.3]], [5, 6])

@testset "add_points!, n-dim" begin
    @test length(d.xs) == 3
end

@testset "mean at point, n-dim" begin
    @test 4.0 <= only(mean(finite_posterior(d, [[4.9, 4.1]]))) <= 5.0
end

@testset "mean at points, n-dim" begin
    @test length(mean(finite_posterior(d, [[2.0, 4.3], [2.4, 1.3]]))) == 2
end

# by default, noise_var is set to 0.1
@testset "var at point, n-dim" begin
    @test 0 <= only(var(finite_posterior(d, [[5.9, 7.0]]))) <= 0.3
end

@testset "var at points, n-dim" begin
    @test all(0 .<= var(finite_posterior(d, [[6.0, 7.1], [5.0, 21.3]])) .<= 0.3)
end

@testset "mean_and_var at point, n-dim" begin
    μ, σ² = only.(mean_and_var(finite_posterior(d, [[4.9, 4.1]])))
    @test 4.0 < μ < 5.0
    @test 0 <= σ²
end

@testset "mean_and_var at points, n-dim" begin
    μs, σ²s = mean_and_var(finite_posterior(d, [[2.0, 4.3], [2.4, 1.3]]))
    @test length(μs) == 2
    @test length(σ²s) == 2
end

@testset "read default hyperparameters, n-dim" begin
    @test hyperparameters(d).noise_var == 0.1
    @test length(keys(hyperparameters(d))) == 1
end

@testset "update_hyperparameters!, n-dim" begin
    old_prior = AbstractGPs.GP(d.kernel_creator(delete(d.hyperparameters, :noise_var)))
    old_logpdf = AbstractGPs.logpdf(old_prior(d.xs, hyperparameters(d).noise_var), d.ys)
    update_hyperparameters!(d, unif_prior)
    new_prior = AbstractGPs.GP(d.kernel_creator(delete(d.hyperparameters, :noise_var)))
    new_logpdf = AbstractGPs.logpdf(new_prior(d.xs, hyperparameters(d).noise_var), d.ys)
    # new hyperparameters should make data more probable, maximize marginal logpdf
    @test old_logpdf <= new_logpdf
end

@testset "rand at point, n-dim" begin
    @test isa(only(rand(finite_posterior(d, [[1.3, 4.0]]))), Number)
end

@testset "rand at points, n-dim" begin
    @test length(rand(finite_posterior(d, [[1.3, 4.0], [5.6, 4.0]]))) == 2
    @test isa(rand(finite_posterior(d, [[1.3, 4.0], [5.6, 4.0]])), Vector{Float64})
end

@testset "utilities" begin
    @test keys(SurrogatesAbstractGPs.delete((; a = 2, b = 4), :a)) == (:b,)
    @test keys(SurrogatesAbstractGPs.delete((; a = 2, b = 4), :c)) == (:a, :b)
end

# 2-dim surrogate
d = GPSurrogate([[5.0, 4.0]], [5])
add_points!(d, [[6.0, 7.1], [5.0, 21.3]], [5, 6])

unif_prior = BoundedHyperparameters([(noise_var = bounded(v, 0.0001, 0.2),)
                                     for v in range(0.00011, 0.1999, length = 10)])
add_points!(d, [[6.0, 7.1], [5.0, 21.3]], [5, 6])

@testset "update_hyperparameters! with multiple restarts, n-dim" begin
    old_prior = AbstractGPs.GP(d.kernel_creator(delete(d.hyperparameters, :noise_var)))
    old_logpdf = AbstractGPs.logpdf(old_prior(d.xs, hyperparameters(d).noise_var), d.ys)
    update_hyperparameters!(d, unif_prior)
    new_prior = AbstractGPs.GP(d.kernel_creator(delete(d.hyperparameters, :noise_var)))
    new_logpdf = AbstractGPs.logpdf(new_prior(d.xs, hyperparameters(d).noise_var), d.ys)
    # new hyperparameters should make data more probable, maximize marginal logpdf
    @test old_logpdf <= new_logpdf
end

# # TODO: integrate old tests from registered SurrogatesAbstractGPs package,
# # https://github.com/SciML/Surrogates.jl/tree/e6aa022e612ac57228506e625c662438d385e69d/lib/SurrogatesAbstractGPs
