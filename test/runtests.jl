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
    @test_throws ErrorException BinaryMetrics([0;1], [0.4;-0.1])
    @test_throws ErrorException BinaryMetrics([0;1;1], [0.4;0.1])

    B = BinaryMetrics([0;0;1;1], [0.4;0.7;0.4;1])
    @test tp(B,0.5) == 1
    @test tn(B,0.5) == 1
    @test fp(B,0.5) == 1
    @test fn(B,0.5) == 1
    @test cm(B,0.5) == ones(2,2)

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

  end

  @testset "testing with ROC of other systems" begin

    # comparing with ROCR
    data = CSV.read(joinpath(@__DIR__, "data", "ROCRdata.csv"))
    scores = data.predictions
    labels = data.labels

    # testing constructors
    curve = ROC(labels, scores)
    B = BinaryMetrics(labels, scores)
    curve = ROC(B)
    println(curve)
    @test isapprox( auc(curve),       0.834187    , atol = 1e-4) # ROCR 0.8341875
    @test isapprox( auc(curve, 0.01), 0.000329615 , atol = 1e-4) # ROCR 0.0003296151
    @test isapprox( auc(curve, 0.1 ), 0.0278062   , atol = 1e-4) # ROCR 0.02780625

    @test op(curve) == curve.OP
    @test auc(curve) == curve.AUC
    @test fpr(curve) == curve.FPR
    @test tpr(curve) == curve.TPR

    # "ROC analysis: web-based calculator for ROC curves' example"
    scores = [1 , 2 , 3 , 4 , 6 , 5 , 7 , 8 , 9 , 10]
    labels = [0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1]
    res = ROC(labels, normalize(scores,1))

    @test auc(res) ≈ 0.96
    @test tpr(res)[1:end-1] == [0.0, 0.2, 0.4, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
    @test fpr(res)[1:end-1] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Example from https://www.epeter-stats.de/roc-curves-and-ties/
    # This package uses the first strategy, i.e.  AUC is equivalent to the
    # Mann-Whitney U statistic

    labels = [ones(Bool,1000); zeros(Bool,1000)]
    scores =  [rand(7:14, 1000); rand(1:8, 1000)]
    roc_data = ROC(labels, normalize(scores,1))
    @test abs( round(auc(roc_data), digits=2) - 0.97 ) ≤ 0.015


    # Example from https://github.com/brian-lau/MatlabAUC/issues/1
    data = [-1 1; 1 2; -1 3; -1 4; 1 5; -1 6; 1 7; -1 8; 1 9; 1 10;
            1 11; -1 13; 1 13; 1 14; 1 14]
    labels = data[:,1] .> 0
    scores = normalize(data[:,2],1)
    roc_data = ROC(labels, scores)
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

    @test norm(round.(reverse(spss[:,1]) - tpr(roc_data)[1:end-1] ,digits=2)) == 0
    @test norm(round.(reverse(spss[:,2]) - fpr(roc_data)[1:end-1] ,digits=2)) == 0

    # Are AUC and ROC consistent after permutation of scores and labels?
    perm = randperm(100)
    scores = rand(100)
    labels = [zeros(50); ones(50)]

    @test auc(ROC(labels, scores)) == auc(ROC(labels[perm], scores[perm]))

  end

  @testset "testing Plots" begin
    # Test ROC
    L = 10
    labels = rand(Bool, L);
    scores = add_noise.(labels, 0.6)
    curve = ROC(labels, scores)
    plot(curve)
  end
end
