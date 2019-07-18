using ROC
using DataFrames
using CSV
using Test
using LinearAlgebra
using Random
Random.seed!(1234)
add_noise(label::Bool, λ=0.0) = label ? 1 - λ*rand() : λ*rand()


@testset "ROC" begin

  @testset "BinClassEval constructors and metrics" begin

    L = 200
    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)

    B = BinClassEval(labels, scores)
    println(B)

    BinClassEval([0;1], [0.4;0.1])
    @test_throws ErrorException BinClassEval([-1;1], [0.4;0.1])
    @test_throws ErrorException BinClassEval([0;1], [0.4;-0.1])
    @test_throws ErrorException BinClassEval([0;1;1], [0.4;0.1])

    B = BinClassEval([0;0;1;1], [0.4;0.7;0.4;1])
    @test tp(B,0.5) == 1
    @test tn(B,0.5) == 1
    @test fp(B,0.5) == 1
    @test fn(B,0.5) == 1
    @test conf_mtx(B,0.5) == ones(2,2)

    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)
    B = BinClassEval(labels, scores)
    points = [0.2;0.5;0.7]
    tps = tp(B, points)
    tns = tn(B, points)
    fps = fp(B, points)
    fns = fn(B, points)
    @test conf_mtx(B,points) == (tps, fps, fns, tns)

    @test isapprox(tpr(B, points) , 1 .- fnr(B,points); atol=1e-8)
    @test isapprox(tpr(B, points) , tps./(tps.+fns); atol=1e-8)
    @test isapprox(tnr(B, points) , 1 .- fpr(B,points); atol=1e-8)
    @test isapprox(tnr(B, points) , tns./(tns.+fps); atol=1e-8)
    @test isapprox(fnr(B, points) , 1 .- tpr(B,points); atol=1e-8)
    @test isapprox(fnr(B, points) , fns./(fns.+tps); atol=1e-8)
    @test isapprox(fpr(B, points) , 1 .- tnr(B,points); atol=1e-8)
    @test isapprox(fpr(B, points) , fps./(fps.+tns); atol=1e-8)

  end

  @testset "testing with ROCs of other systems" begin

    # comparing with ROCR
    data = CSV.read(joinpath(@__DIR__, "data", "ROCRdata.csv"))
    scores = data.predictions
    labels = data.labels

    curve = ROCs(labels, scores)
    @test isapprox( auc(curve),       0.834187    , atol = 1e-4) # ROCR 0.8341875
    @test isapprox( auc(curve, 0.01), 0.000329615 , atol = 1e-4) # ROCR 0.0003296151
    @test isapprox( auc(curve, 0.1 ), 0.0278062   , atol = 1e-4) # ROCR 0.02780625


    # "ROC analysis: web-based calculator for ROC curves' example"
    scores = [1 , 2 , 3 , 4 , 6 , 5 , 7 , 8 , 9 , 10]
    labels = [0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1]
    res = ROCs(labels, normalize(scores,1))

    @test auc(res) ≈ 0.96
    @test res.tpr[1:end-1] == [0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
    @test res.fpr[1:end-1] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Example from https://www.epeter-stats.de/roc-curves-and-ties/
    # This package uses the first strategy, i.e.  AUC is equivalent to the
    # Mann-Whitney U statistic

    labels = [ones(Bool,1000); zeros(Bool,1000)]
    scores =  [rand(7:14, 1000); rand(1:8, 1000)]
    roc_data = ROCs(labels, normalize(scores,1))
    @test abs( round(auc(roc_data), digits=2) - 0.97 ) ≤ 0.015


    # Example from https://github.com/brian-lau/MatlabAUC/issues/1
    data = [-1 1; 1 2; -1 3; -1 4; 1 5; -1 6; 1 7; -1 8; 1 9; 1 10;
            1 11; -1 13; 1 13; 1 14; 1 14]
    labels = data[:,1] .> 0
    scores = normalize(data[:,2],1)
    roc_data = ROCs(labels, scores)
    spss = [1.000 1.000
            1.000 .833
            .889 .833
            .889 .667
            .889 .500
            .778 .500
            .778 .333
            .667 .333
            .667 .167
            .556 .167
            .444 .167
            .333 .167
            .222 .000
            .000 .000]

    @test norm(round.(reverse(spss[:,1]) - roc_data.tpr[1:end-1] ,digits=2)) == 0
    @test norm(round.(reverse(spss[:,2]) - roc_data.fpr[1:end-1] ,digits=2)) == 0

    # Are AUC and ROC consistent after permutation of scores and labels?
    perm = randperm(100)
    scores = rand(100)
    labels = [zeros(50); ones(50)]

    @test auc(ROCs(labels, scores)) == auc(ROCs(labels[perm], scores[perm]))

  end
