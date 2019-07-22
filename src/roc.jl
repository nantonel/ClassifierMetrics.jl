export ROC, auc

struct ROC{T <: Real} <: AbstractBinaryMetric
  labels::BitVector
  scores::Vector{T}
  AUC::T
  FPR::Vector{T}
  TPR::Vector{T}
  OP::Vector{T}
  function ROC(labels,scores)
    B = BinaryMetrics(labels, scores)
    OP = op(B)
    FPR = fpr(B, OP)
    TPR = tpr(B, OP)
    AUC = roc_auc(FPR, TPR, OP)
    new{eltype(FPR)}(B.labels, B.scores, AUC, FPR, TPR, OP)
  end
end

function Base.show(io::IO, B::ROC) 
  println(io,"Receiver Operating Characteristic (ROC) curve")
  AUC = round(auc(B),digits=2)
  print(io,"Samples:$(length(B)), P:$(positives(B)), N:$(negatives(B)), AUC:$(AUC)")
end

"""
  ROC(labels, scores)

Generates data for plotting a Receiver Operating Characteristic (ROC) curve.

* `labels` must be a boolean array e.g. either containing `true`,`false` or `0`,`1`.
* `scores` must be an array with the same length of `labels` containing the scores of the binary classifier. Scores must be normalized such that they are between 0 and 1.

You can generate a plot automatically using the following commands:
```julia
roc = ROC(labels, scores)

using Plots
plot(roc)
```

The Area Under Curve (AUC) FPR and TPR can be accessed by `auc(roc)`, `fpr(roc)` and `tpr(roc)` respectively.

"""
ROC(B::BinaryMetrics) = ROC(B.labels, B.scores)

function roc_auc(FPR::Vector{T}, TPR::Vector{T}, OP::Vector{T}) where {T}
  AUC=zero(T)
  for i in 2:length(OP)
    dx = FPR[i] - FPR[i-1]
    dy = TPR[i] - TPR[i-1]
    AUC += ( (dx*TPR[i-1]) + (0.5*dx*dy) )
  end
  return AUC
end

auc(curve::ROC) = curve.AUC
fpr(curve::ROC) = curve.FPR
tpr(curve::ROC) = curve.TPR
op(curve::ROC) = curve.OP
