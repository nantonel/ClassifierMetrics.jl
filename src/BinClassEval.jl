export BinClassEval

struct BinClassEval{T <: Real}
  labels::BitVector
  scores::Vector{T}

  function BinClassEval(labels::AbstractVector, scores::AbstractVector)
    L = length(labels)

    if length(scores) != L
      throw(ErrorException("scores and labels should have the same length."))
    end
    if !all(0 .<= scores .<= 1)
      throw(ErrorException("scores must be between 0 and 1"))
    end
    if sum(labels .== true) + sum(labels.==false) != L
      throw(ErrorException("Labels must be either `true` or `false`  (`0` or `1`)"))
    end
    labels = BitArray(labels)

    new{eltype(scores)}(labels, scores)
  end
end

Base.length(B::BinClassEval) = length(B.labels)
positives(B::BinClassEval) = sum(B.labels)
negatives(B::BinClassEval) = length(B) - positives(B)

function Base.show(io::IO, B::BinClassEval) 
  println(io,"Evaluation of Binary Classifier with $(length(B)) samples")
  print(io,"P: $(positives(B)), N:$(negatives(B))")
end

export positives, negatives, tp, tn, fp, fn, conf_mtx 
export tpr, tnr, fpr, fnr
import Base.broadcast

is_tp(label, score, point) =  label && score >= point
is_tn(label, score, point) = !label && score  < point
is_fp(label, score, point) = !label && score >= point
is_fn(label, score, point) =  label && score  < point

for f in [
          :tp => :is_tp,
          :tn => :is_tn,
          :fp => :is_fp,
          :fn => :is_fn,
         ] 
  @eval begin
    function $(f[1])(B::BinClassEval, point::Real)
      cnt = 0
      for i in eachindex(B.scores)
        if $(f[2])(B.labels[i], B.scores[i], point) 
          cnt += 1 
        end
      end
      return cnt
    end

    function $(f[1])(B::BinClassEval, points::AbstractArray)
      cnts = zero(points)
      for i in eachindex(points)
        cnts[i] = $(f[1])(B, points[i])
      end
      return cnts
    end

  end
end

"""
  tp(B::BinClassEval, point)

Returns the true positives of `B` at a particular operational point.
"""
tp

"""
  tn(B::BinClassEval, point)

Returns the true negatives of `B` at a particular operational point.
"""
tn

"""
  fp(B::BinClassEval, point)

Returns the false positives of `B` at a particular operational point.
"""
fp

"""
  fn(B::BinClassEval, point)

Returns the false negatives of `B` at a particular operational point.
"""
fn

"""
  tpr(B::BinClassEval, point)

Returns the true positive rate of `B` at a particular operational point.
"""
tpr(B::BinClassEval,point) = tp(B,point) ./ positives(B)

"""
  tnr(B::BinClassEval, point)

Returns the true negative rate of `B` at a particular operational point.
"""
tnr(B::BinClassEval,point) = tn(B,point) ./ negatives(B)

"""
  fnr(B::BinClassEval, point)

Returns the false negative rate of `B` at a particular operational point.
"""
fnr(B::BinClassEval,point) = fn(B,point) ./ positives(B)

"""
  fpr(B::BinClassEval, point)

Returns the false positive rate of `B` at a particular operational point.
"""
fpr(B::BinClassEval,point) = fp(B,point) ./ negatives(B)


"""
  conf_mtx(B::BinClassEval, point::Real)

Returns the confusion matrix of `B` at a given operational point.
The matrix is defined as follow:

| true positives  | false positives | 
| false negatives | true negatives  | 
"""
conf_mtx(B::BinClassEval, point::Real) =  [ tp(B,point) fp(B,point); 
                                            fn(B,point) tn(B,point) ]

"""
  conf_mtx(B::BinClassEval, point::Array)

Returns the confusion matrix of `B`. The output is conveniently given as:

`(tps, fps, fns, tns)`

Each element of the Tuple is an array of `length(point)`.
"""
conf_mtx(B::BinClassEval, point::AbstractArray) =  (tp(B,point), fp(B,point), 
                                                    fn(B,point), tn(B,point))
export op_points

"""
  op_points(B::BinClassEval)

Returns the significant operational points of `B`.
"""
function op_points(B::BinClassEval)

  thresholds = sort(unique(B.scores); rev=true) #threshold
  if thresholds[1] != 1.0 pushfirst!(thresholds,1.0) end
  if thresholds[end] != 0 push!(thresholds,0.0) end

  return thresholds

end

