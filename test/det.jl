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
