using SurrogatesAbstractGPs
using Test
using ParameterHandling
using KernelFunctions

unif_prior = BoundedHyperparameters((;
    lengthscale = bounded(1.0, 0.004, 4.0),
    noise_var = bounded(0.1, 0.0001, 0.2)))

function kernel_creator(hyperparameters)
    with_lengthscale(KernelFunctions.Matern52Kernel(), hyperparameters.lengthscale)
end

s = GPSurrogate([1.0, 3.0, 4.0], [0.9, 3.1, 4.5],
    kernel_creator = kernel_creator,
    hyperparameters = (; lengthscale = 1.0, noise_var = 0.1))

@testset "add_point!, add_points!" begin
    add_point!(s, 5.0, 5.9)
    add_points!(s, [6.0, 7.1], [5.9, 6.9])
    @test length(s.x) == 6
    @test all(s.x .== [1.0, 3.0, 4.0, 5.0, 6.0, 7.1])
    @test all(s.y .== [0.9, 3.1, 4.5, 5.9, 5.9, 6.9])
    # add 2d points with Int objective values
    d = GPSurrogate(Vector{Vector{Float64}}(), Vector{Int}())
    add_point!(d, [5.0, 4.0], 5)
    add_points!(d, [[6.0, 7.1], [5.0, 21.3]], [5, 6])
    @test length(d.x) == 3
end

@testset "mean_at_point" begin
    @test 1 < mean_at_point(s, 2.0) < 3
end

@testset "std_error_at_point" begin
    @test 0 <= std_error_at_point(s, 1.0) <= 1
end

@testset "hyperparameters" begin
    @test hyperparameters(s).lengthscale == 1.0
    @test hyperparameters(s).noise_var == 0.1
    @test length(keys(hyperparameters(s))) == 2
end

@testset "logpdf_surrogate, update_hyperparameters!" begin
    old_logpdf = logpdf_surrogate(s)
    update_hyperparameters!(s, unif_prior)
    # new hyperparameters should make data more probable, maximize marginal logpdf
    @test old_logpdf <= logpdf_surrogate(s)
end

@testset "supports_posterior, supports_hyperparameters" begin
    @test supports_posterior(s)
    @test supports_hyperparameters(s)
end

@testset "joint posterior" begin
    posterior(s, [2.3])
    posterior(s, [2.4, 5.43, 5.0])
end

@testset "utilities" begin
    @test keys(SurrogatesAbstractGPs.delete( (; a=2,b=4), :a)) == (:b,)
    @test keys(SurrogatesAbstractGPs.delete( (; a=2,b=4), :c)) == (:a, :b)
end

# TODO: add old tests
