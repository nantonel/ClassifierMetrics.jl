using ClassifierMetrics
using DataFrames
using CSV
using Test
using LinearAlgebra
using Random
using Plots
Random.seed!(1234)

#utils
add_noise(label::Bool, λ=0.0) = label ? 1 - λ*rand() : λ*rand()

@testset "ClassifierMetrics" begin
  @testset "Binrary Metrics" begin
    include("binarymetrics.jl")
  end
  @testset "testing ROC" begin
    include("roc.jl")
  end
  @testset "testing PR" begin
    include("pr.jl")
  end
  @testset "testing DET" begin
    include("det.jl")
  end
  @testset "testing Plots" begin
    include("plots.jl")
  end
end
