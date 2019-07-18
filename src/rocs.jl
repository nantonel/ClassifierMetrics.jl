export ROCs, auc

struct ROCs{T <: Real}
  B::BinClassEval{T}
  fpr::Vector{T}
  tpr::Vector{T}
  op_points::Vector{T}
  function ROCs(labels,scores)
    B = BinClassEval(labels,scores)
    points = op_points(B)
    fprs = fpr(B,points)
    tprs = tpr(B,points)
    new{eltype(fprs)}(B, fprs, tprs, points)
  end
end

function auc(roc::ROCs{T}) where {T}
  auc=zero(T)

  for i in 2:length(roc.op_points)
    dx = roc.fpr[i] - roc.fpr[i-1]
    dy = roc.tpr[i] - roc.tpr[i-1]
    auc += ( (dx*roc.tpr[i-1]) + (0.5*dx*dy) )
  end
  return auc
end

function auc(roc::ROCs{T}, FPRstop::T) where {T}
	auc=zero(T)
	if FPRstop <= 0 || FPRstop >= 1
		error("FPRstop should be in (0,1)")
	end
	for i in 2:length(roc.op_points)
		if roc.fpr[i] > FPRstop
			dx = roc.fpr[i] - roc.fpr[i-1]
			dy = roc.tpr[i] - roc.tpr[i-1]
			dxstop = FPRstop - roc.fpr[i-1]
			dystop = (dy/dx)*dxstop
			auc += ( (dxstop*roc.tpr[i-1]) + (0.5*dxstop*dystop) )
			break
		end
		dx = roc.fpr[i] - roc.fpr[i-1]
		dy = roc.tpr[i] - roc.tpr[i-1]
		auc += ( (dx*roc.tpr[i-1]) + (0.5*dx*dy) )
	end
	return auc
end
