using SurrogatesAbstractGPs
using Test
using ParameterHandling
using KernelFunctions
import AbstractGPs

unif_prior = BoundedHyperparameters((;
    lengthscale = bounded(1.0, 0.004, 4.0),
    noise_var = bounded(0.1, 0.0001, 0.2)))

function kernel_creator(hyperparameters)
    with_lengthscale(KernelFunctions.Matern52Kernel(), hyperparameters.lengthscale)
end

# 1-dim surrogate
s = GPSurrogate([1.0, 3.0, 4.0], [0.9, 3.1, 4.5],
    kernel_creator = kernel_creator,
    hyperparameters = (; lengthscale = 1.0, noise_var = 0.1))

@testset "add_point!, 1-dim" begin
    add_point!(s, 5.0, 5.9)
    add_point!(s, [6.0, 7.1], [5.9, 6.9])
    @test length(s.xs) == 6
    @test all(s.xs .== [1.0, 3.0, 4.0, 5.0, 6.0, 7.1])
    @test all(s.ys .== [0.9, 3.1, 4.5, 5.9, 5.9, 6.9])
end

@testset "mean at point, 1-dim" begin
    @test 1 < mean(s, 2.0) < 3
end

@testset "mean at points, 1-dim" begin
    @test length(mean(s, [2.0, 4.3])) == 2
end

@testset "var at point, 1-dim" begin
    @test 0 <= var(s, 1.0) <= 1
end

@testset "var at points, 1-dim" begin
    @test length(var(s, [2.0, 4.3, 0.4])) == 3
end

@testset "read hyperparameters, 1-dim" begin
    @test hyperparameters(s).lengthscale == 1.0
    @test hyperparameters(s).noise_var == 0.1
    @test length(keys(hyperparameters(s))) == 2
end

@testset "update_hyperparameters!, 1-dim" begin
    old_logpdf = AbstractGPs.logpdf(s.gp(s.xs, hyperparameters(s).noise_var), s.ys)
    update_hyperparameters!(s, unif_prior)
    new_logpdf = AbstractGPs.logpdf(s.gp(s.xs, hyperparameters(s).noise_var), s.ys)
    # new hyperparameters should make data more probable, maximize marginal logpdf
    @test old_logpdf <= new_logpdf
end

@testset "joint posterior, 1-dim" begin
    posterior(s, [2.3])
    posterior(s, [2.4, 5.43, 5.0])
end

@testset "rand at point, 1-dim" begin
    @test isa(rand(s, 1.3), Number)
end

@testset "rand at points, 1-dim" begin
    @test length(rand(s, [1.3, 4.]))== 2
end

# 2-dim surrogate
d = GPSurrogate(Vector{Vector{Float64}}(), Vector{Int}())
add_point!(d, [5.0, 4.0], 5)
add_point!(d, [[6.0, 7.1], [5.0, 21.3]], [5, 6])

@testset "add_point!, n-dim" begin
    @test length(d.xs) == 3
end

@testset "mean at point, n-dim" begin
    @test 4. <= mean(d, [4.9, 4.1]) <= 5.
end

@testset "mean at points, n-dim" begin
    @test length(mean(d, [[2.0, 4.3],[2.4, 1.3]])) == 2
end

# by default, noise_var is set to 0.1
@testset "var at point, n-dim" begin
    @test 0 <= var(d, [5.9, 7.0]) <= 0.3
end

@testset "var at points, n-dim" begin
    @test all(0 .<= var(d, [[6.0, 7.1], [5.0, 21.3]]) .<= 0.3)
end

@testset "read default hyperparameters, n-dim" begin
    @test hyperparameters(d).noise_var == 0.1
    @test length(keys(hyperparameters(d))) == 1
end

@testset "update_hyperparameters!, n-dim" begin
    old_logpdf = AbstractGPs.logpdf(d.gp(d.xs, hyperparameters(d).noise_var), d.ys)
    update_hyperparameters!(d, unif_prior)
    new_logpdf = AbstractGPs.logpdf(d.gp(d.xs, hyperparameters(d).noise_var), d.ys)
    # new hyperparameters should make data more probable, maximize marginal logpdf
    @test old_logpdf <= new_logpdf
end

@testset "joint posterior, n-dim" begin
    posterior(d, [2.3, 5.])
    posterior(d, [[2.4, 5.43],[3.,5.]])
end
@testset "rand at point, n-dim" begin
    @test isa(rand(d, [1.3, 4.]), Number)
end

@testset "rand at points, n-dim" begin
    @test length(rand(d, [[1.3, 4.], [5.6,4.]]))== 2
    @test isa(rand(d, [[1.3, 4.], [5.6,4.]]), Vector{Float64})
end

@testset "utilities" begin
    @test keys(SurrogatesAbstractGPs.delete((; a = 2, b = 4), :a)) == (:b,)
    @test keys(SurrogatesAbstractGPs.delete((; a = 2, b = 4), :c)) == (:a, :b)
end

# TODO: integrate old tests from registered SurrogatesAbstractGPs package,
# https://github.com/SciML/Surrogates.jl/tree/e6aa022e612ac57228506e625c662438d385e69d/lib/SurrogatesAbstractGPs
