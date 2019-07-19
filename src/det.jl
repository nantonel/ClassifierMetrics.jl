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
    new{eltype(FPR)}(B.labels, B.scores, FNR, FPR, OP)
  end
end

DET(B::BinaryMetrics) = DET(B.labels, B.scores)

auc(curve::DET) = curve.AUC
fpr(curve::DET) = curve.FPR
fnr(curve::DET) = curve.FNR
op(curve::DET) = curve.OP
