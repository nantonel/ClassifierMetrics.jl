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

OP = op(B)
@test op(curve) == OP 
@test auc(curve) == curve.AUC
@test fpr(curve) == fpr(B,OP)
@test tpr(curve) == tpr(B,OP)

# "ROC analysis: web-based calculator for ROC curves' example"
scores = [1 , 2 , 3 , 4 , 6 , 5 , 7 , 8 , 9 , 10]
labels = [0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1]
res = ROC(labels, scores)

@test auc(res) ≈ 0.96
@test tpr(res) == [0.2, 0.4, 0.6, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0]
@test fpr(res) == [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]

# Example from https://www.epeter-stats.de/roc-curves-and-ties/
# This package uses the first strategy, i.e.  AUC is equivalent to the
# Mann-Whitney U statistic

labels = [ones(Bool,1000); zeros(Bool,1000)]
scores =  [rand(7:14, 1000); rand(1:8, 1000)]
roc_data = ROC(labels, scores)
@test abs( round(auc(roc_data), digits=2) - 0.97 ) ≤ 0.015


# Example from https://github.com/brian-lau/MatlabAUC/issues/1
data = [-1 1; 1 2; -1 3; -1 4; 1 5; -1 6; 1 7; -1 8; 1 9; 1 10;
        1 11; -1 13; 1 13; 1 14; 1 14]
labels = data[:,1] .> 0
scores = data[:,2]
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

@test norm(round.(reverse(spss[:,1])[2:end] - tpr(roc_data), digits=2)) == 0
@test norm(round.(reverse(spss[:,2])[2:end] - fpr(roc_data), digits=2)) == 0

# Are AUC and ROC consistent after permutation of scores and labels?
perm = randperm(100)
scores = rand(100)
labels = [zeros(50); ones(50)]

@test auc(ROC(labels, scores)) == auc(ROC(labels[perm], scores[perm]))
