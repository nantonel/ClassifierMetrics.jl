export BinaryMetrics

abstract type AbstractBinaryMetric end

struct BinaryMetrics{T <: Real} <: AbstractBinaryMetric
  labels::BitVector
  scores::Vector{T}

  function BinaryMetrics(labels::AbstractVector, scores::AbstractVector)
    L = length(labels)
    scores = float(scores)

    if length(scores) != L
      throw(ErrorException("scores and labels should have the same length."))
    end
    if sum(labels .== true) + sum(labels.==false) != L
      throw(ErrorException("Labels must be either `true` or `false`  (`0` or `1`)"))
    end
    labels = BitArray(labels)
    scores = Array{eltype(scores)}(scores)

    new{eltype(scores)}(labels, scores)
  end
end


"""
  `BinaryMetrics(labels, scores)`

Create an object to compute various metrics of a binary classifier.

For example False Positive Rate at a particular operating point `OP` can be obtained by:
```julia
B = BinaryMetrics(labels, scores)
fpr(B,OP)

```
See documentation for a comprehensive list of available metrics.

"""
BinaryMetrics

Base.length(B::AbstractBinaryMetric) = length(B.labels)
positives(B::AbstractBinaryMetric) = sum(B.labels)
negatives(B::AbstractBinaryMetric) = length(B) - positives(B)

function Base.show(io::IO, B::AbstractBinaryMetric) 
  println(io,"Binary Classifier Results ")
  print(io,"Samples:$(length(B)), P:$(positives(B)), N:$(negatives(B))")
end

export positives, negatives
export tp, tn, fp, fn, cm 
export tpr, recall, tnr, fpr, fnr
export ppv, fdr
export npv, false_omission_rate, csi
export acc, f1, mcc
export eer, fpr_at_tpr

is_tp(label, score, OP) =  label && score >= OP 
is_tn(label, score, OP) = !label && score  < OP 
is_fp(label, score, OP) = !label && score >= OP 
is_fn(label, score, OP) =  label && score  < OP 

for f in [
          :tp => :is_tp,
          :tn => :is_tn,
          :fp => :is_fp,
          :fn => :is_fn,
         ] 
  @eval begin
    function $(f[1])(B::AbstractBinaryMetric, OP::Real)
      cnt = zero(Int)
      for i in eachindex(B.scores)
        if $(f[2])(B.labels[i], B.scores[i], OP) 
          cnt += 1 
        end
      end
      return cnt
    end

    function $(f[1])(B::AbstractBinaryMetric, OP::AbstractVector)
      cnts = zeros(Int,length(OP))
      for i in eachindex(OP)
        cnts[i] = $(f[1])(B, OP[i])
      end
      return cnts
    end

  end
end

"""
  `tp(B::AbstractBinaryMetric, OP=op(B))`

Returns the true positives of `B` at a particular operating point.
"""
tp

"""
  `tn(B::AbstractBinaryMetric, OP=op(B))`

Returns the true negatives of `B` at a particular operating point.
"""
tn

"""
  `fp(B::AbstractBinaryMetric, OP=op(B))`

Returns the false positives of `B` at a particular operating point.
"""
fp

"""
  `fn(B::AbstractBinaryMetric, OP=op(B))`

Returns the false negatives of `B` at a particular operating point.
"""
fn

"""
  `tpr(B::AbstractBinaryMetric, OP=op(B))`

Returns the true positive rate (aka recall) of `B` at a particular operating point.
"""
tpr(B::AbstractBinaryMetric, OP=op(B)) = tp(B,OP) ./ positives(B)
recall = tpr

"""
  `tnr(B::AbstractBinaryMetric, OP=op(B))`

Returns the true negative rate of `B` at a particular operating point.
"""
tnr(B::AbstractBinaryMetric, OP=op(B)) = tn(B,OP) ./ negatives(B)

"""
  `fnr(B::AbstractBinaryMetric, OP=op(B))`

Returns the false negative rate of `B` at a particular operating point.
"""
fnr(B::AbstractBinaryMetric,OP=op(B)) = fn(B,OP) ./ positives(B)

"""
  `fpr(B::AbstractBinaryMetric, OP=op(B))`

Returns the false positive rate of `B` at a particular operating point.
"""
fpr(B::AbstractBinaryMetric,OP=op(B)) =fp(B,OP) ./ negatives(B)

"""
  `ppv(B::AbstractBinaryMetric, OP=op(B))`

Returns the positive predictive value (aka precision) of `B` at a particular operating point.
"""
function ppv(B::AbstractBinaryMetric,OP=op(B)) 
  TP = tp(B,OP)
  FP = fp(B,OP)
  return TP ./ ( TP .+ FP )
end

import Base: precision
precision(B::AbstractBinaryMetric,OP=op(B)) = ppv(B,OP)

"""
  `fdr(B::AbstractBinaryMetric, OP=op(B))`

Returns the false discovery rate of `B` at a particular operating point.
"""
function fdr(B::AbstractBinaryMetric,OP=op(B)) 
  TP = tp(B,OP)
  FP = fp(B,OP)
  return FP ./ ( TP .+ FP )
end

"""
  `npv(B::AbstractBinaryMetric, OP=op(B))`

Returns the negative predictive value of `B` at a particular operating point.
"""
function npv(B::AbstractBinaryMetric, OP=op(B)) 
  TN = tn(B,OP)
  FN = fn(B,OP)
  return TN ./ ( TN .+ FN )
end

"""
  `false_omission_rate(B::AbstractBinaryMetric, OP=op(B))`

Returns the false omission rate of `B` at a particular operating point.
"""
function false_omission_rate(B::AbstractBinaryMetric, OP=op(B)) 
  TN = tn(B,OP)
  FN = fn(B,OP)
  return FN ./ ( TN .+ FN )
end

"""
  `csi(B::AbstractBinaryMetric, OP=op(B))`

Returns the critical success index of `B` at a particular operating point.
"""
function csi(B::AbstractBinaryMetric, OP=op(B)) 
  TP = tp(B,OP)
  FN = fn(B,OP)
  FP = fp(B,OP)
  return TP ./ ( TP .+ FN .+ FP )
end

"""
  `acc(B::AbstractBinaryMetric, OP=op(B))`

Returns the accuracy index of `B` at a particular operating point.
"""
function acc(B::AbstractBinaryMetric, OP=op(B)) 
  TP = tp(B,OP)
  TN = tn(B,OP)
  return (TP + TN) ./ ( length(B) )
end

"""
  `f1(B::AbstractBinaryMetric, OP=op(B))`

Returns the F1 score of `B` at a particular operating point.
"""
function f1(B::AbstractBinaryMetric, OP=op(B)) 
  PPV = ppv(B, OP)
  TPR = tpr(B, OP)
  return 2 .*(PPV .* TPR) ./ ( PPV .+ TPR )
end

"""
  `mcc(B::AbstractBinaryMetric, OP=op(B))`

Returns the Matthews correlation coefficient of `B` at a particular operating point.
"""
function mcc(B::AbstractBinaryMetric, OP=op(B)) 
  TP, FP, FN, TN = tp(B,OP), fp(B,OP), fn(B,OP), tn(B,OP)
  return  @. (TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
end

"""
  `cm(B::AbstractBinaryMetric, OP::Real)`

Returns the confusion matrix of `B` at a given operating point.
The matrix is defined as follow:

| true positives  | false positives | 
| false negatives | true negatives  | 
"""
cm(B::AbstractBinaryMetric, OP::Real) =  [ tp(B,OP) fp(B,OP); 
                                          fn(B,OP) tn(B,OP) ]

"""
  `cm(B::AbstractBinaryMetric, OP::Array)`

Returns the confusion matrix of `B`. The output is conveniently given as:

`(tps, fps, fns, tns)`

Each element of the Tuple is an array of `length(OP)`.
"""
cm(B::AbstractBinaryMetric, OP::AbstractArray) =  (tp(B,OP), fp(B,OP), 
                                                   fn(B,OP), tn(B,OP))
export op

"""
  `op(B::AbstractBinaryMetric)`

Returns significant operating points of `B`.
"""
function op(B::AbstractBinaryMetric)

  OP = sort(unique(B.scores); rev=true) #threshold

  return OP

end


"""
  `eer(FPR::Array,FNR::Array) -> (EER, index)`

Returns the Equal Error Rate (EER) from the intersection of the False Positive rate (must be monotonically increasing) and the False Negative Rate (must be monotonically increasing) and the index of this intersection.

"""
function eer(FPR,FNR)
  if length(FPR) != length(FNR)
    throw(ErrorException("FPR and FNR must have the same length."))
  end
  if !issorted(FPR) 
    throw(ErrorException("FPR must be monotonically decreasing."))
  end
  if !issorted(FNR,rev=true)
    throw(ErrorException("FNR must be monotonically increasing."))
  end
  idx = findfirst(FPR .>= FNR)
  EER = (FPR[idx]+FNR[idx])/2
  return EER, idx 
end

"""
  `eer(B::AbstractBinaryMetric)`

Returns the Equal Error Rate of `B`. 
"""
function eer(B::AbstractBinaryMetric)
  OP = op(B)
  FPR = fpr(B,OP)
  FNR = fnr(B,OP)
  eer(FPR,FNR)
end

"""
  `fpr_at_tpr(B::AbstractBinaryMetric; tpr_at=0.95)`

Returns the false positive rate of `B` when the true positive rate is at `tpr_at`. 
"""
function fpr_at_tpr(B::AbstractBinaryMetric, tpr_at=0.95)
  idx = findfirst(tpr(B) .>= tpr_at)
  return fpr(B)[idx]
end
