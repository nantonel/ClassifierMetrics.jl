L = 20
labels = rand(Bool, L);
scores = add_noise.(labels, 0.6)

curve = PR(labels, scores)
B = BinaryMetrics(labels, scores)
curve = PR(B)
println(curve)

OP = op(B)
@test op(curve) == OP 
@test precision(curve) == precision(B,OP)
@test recall(curve) == tpr(B,OP)

# comparing with sklearn
data = CSV.read(joinpath(@__DIR__, "data", "data.csv"))
pr_sklearn = CSV.read(joinpath(@__DIR__, "data", "pr_sklearn.csv"))

precision_sklearn = pr_sklearn[:,:precision]
recall_sklearn = pr_sklearn[:,:recall]

curve = PR(data.labels, data.predictions)
@test norm(recall(curve)-reverse(recall_sklearn)[2:end]) <= 1e-8
@test norm(precision(curve)-reverse(precision_sklearn)[2:end]) <= 1e-8
@test auc(curve)-0.7814245832778571 < 1e-8
