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

  @testset "BinaryMetrics constructors and metrics" begin

    L = 200
    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)

    B = BinaryMetrics(labels, scores)
    println(B)

    BinaryMetrics([0;1], [0.4;0.1])
    @test_throws ErrorException BinaryMetrics([-1;1], [0.4;0.1])
    @test_throws ErrorException BinaryMetrics([0;1;1], [0.4;0.1])

    B = BinaryMetrics([0;0;1;1], [0.4;0.7;0.4;1])
    @test tp(B,0.5) == 1
    @test tn(B,0.5) == 1
    @test fp(B,0.5) == 1
    @test fn(B,0.5) == 1
    @test cm(B,0.5) == ones(2,2)

    L = 200
    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)
    B = BinaryMetrics(labels, scores)
    OP = [0.2;0.5;0.7]
    tps = tp(B, OP)
    tns = tn(B, OP)
    fps = fp(B, OP)
    fns = fn(B, OP)
    @test cm(B,OP) == (tps, fps, fns, tns)

    @test isapprox(tpr(B, OP) , 1 .- fnr(B,OP); atol=1e-8)
    @test isapprox(tpr(B, OP) , tps./(tps.+fns); atol=1e-8)
    @test isapprox(tnr(B, OP) , 1 .- fpr(B,OP); atol=1e-8)
    @test isapprox(tnr(B, OP) , tns./(tns.+fps); atol=1e-8)
    @test isapprox(fnr(B, OP) , 1 .- tpr(B,OP); atol=1e-8)
    @test isapprox(fnr(B, OP) , fns./(fns.+tps); atol=1e-8)
    @test isapprox(fpr(B, OP) , 1 .- tnr(B,OP); atol=1e-8)
    @test isapprox(fpr(B, OP) , fps./(fps.+tns); atol=1e-8)
    @test isapprox(ppv(B, OP) , 1 .- fdr(B, OP); atol=1e-8)
    @test isapprox(npv(B, OP) , 1 .- false_omission_rate(B, OP); atol=1e-8)
    @test csi(B, OP) == tp(B,OP) ./ (tp(B,OP) + fn(B,OP) +fp(B,OP)) 
    @test acc(B, OP) == (tp(B,OP)+tn(B,OP)) ./ ( tp(B,OP) + tn(B,OP) 
                                                            +fp(B,OP) + fn(B,OP) ) 
    F1 = (2 .* tp(B,OP)) ./ (2 .* tp(B,OP) .+ fp(B,OP) .+ fn(B,OP))
    @test isapprox(f1(B,OP), F1 ; atol=1e-8)
    MCC = (tp(B,OP).*tn(B,OP).-fp(B,OP).*fn(B,OP))./ # <- determinant of CM
    sqrt.((tp(B,OP).+fp(B,OP)).*(tp(B,OP).+fn(B,OP)).*(tn(B,OP).+fp(B,OP)).*(tn(B,OP).+fn(B,OP)))
    @test mcc(B,OP) == MCC 
    @test eer(B) == (0.17,100)  

  end

  @testset "testing with ROC of other systems" begin
    include("roc_tests.jl")
  end

  @testset "testinf DET" begin
    L = 20
    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)

    curve = DET(labels, scores)
    B = BinaryMetrics(labels, scores)
    curve = DET(B)
    println(curve)

    OP = op(B)
    @test op(curve) == OP 
    @test fpr(curve) == fpr(B,OP)
    @test tpr(curve) == tpr(B,OP)
  end

  @testset "testing Plots" begin
    # Test ROC
    L = 10
    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)
    curve = ROC(labels, scores)
    plot(curve)

    curve = DET(labels, scores)
    plot(curve)

  end
end
