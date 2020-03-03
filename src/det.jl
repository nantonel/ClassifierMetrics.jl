export DET

struct DET{T <: Real} <: AbstractBinaryMetric
  labels::BitVector
  scores::Vector{T}
  FPR::Vector{T}
  FNR::Vector{T}
  OP::Vector{T}
  function DET(labels,scores)
    B = BinaryMetrics(labels, scores)
    OP = op(B)
    FNR = fnr(B, OP)
    FPR = fpr(B, OP)
    new{eltype(FPR)}(B.labels, B.scores, FPR, FNR, OP)
  end
end

"""
  DET(labels, scores)

Generates data for plotting a Detection Error Tradeoff (DET) curve.

* `labels` must be a boolean array e.g. either containing `true`,`false` or `0`,`1`.
* `scores` must be an array with the same length of `labels` containing the scores of the binary classifier.

You can generate a plot automatically using the following commands:
```julia
det = DET(labels, scores)

using Plots
plot(det)
```

"""
DET(B::BinaryMetrics) = DET(B.labels, B.scores)

fpr(curve::DET) = curve.FPR
fnr(curve::DET) = curve.FNR
op(curve::DET) = curve.OP
