export PR, auc

struct PR{T <: Real} <: AbstractBinaryMetric
  labels::BitVector
  scores::Vector{T}
  AUC::T
  precision::Vector{T}
  recall::Vector{T}
  OP::Vector{T}
  function PR(labels,scores)
    B = BinaryMetrics(labels, scores)
    OP = op(B)
    p = precision(B, OP)
    r = recall(B,OP)
    AUC = get_auc(r, p)
    new{eltype(p)}(B.labels, B.scores, AUC, p, r, OP)
  end
end

function Base.show(io::IO, B::PR) 
  println(io,"Precision-Recall (PR) curve")
  AUC = round(auc(B),digits=2)
  print(io,"Samples:$(length(B)), P:$(positives(B)), N:$(negatives(B)), AUC:$(AUC)")
end

"""
  PR(labels, scores)

Generates data for plotting a precision-recall (PR) curve.

* `labels` must be a boolean array e.g. either containing `true`,`false` or `0`,`1`.
* `scores` must be an array with the same length of `labels` containing the scores of the binary classifier.

You can generate a plot automatically using the following commands:
```julia
pr = PR(labels, scores)

using Plots
plot(pr)
```

The Area Under Curve (AUC) FPR and TPR can be accessed by `auc(pr)`, `fpr(pr)` and `tpr(pr)` respectively.

"""
PR(B::BinaryMetrics) = PR(B.labels, B.scores)

auc(curve::PR) = curve.AUC
ppv(curve::PR) = curve.precision
tpr(curve::PR) = curve.recall
op(curve::PR) = curve.OP

