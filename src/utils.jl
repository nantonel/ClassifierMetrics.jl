function get_auc(x::Vector{T}, y::Vector{T}) where {T}
  AUC=zero(T)
  for i in 2:length(x)
    dx = x[i] - x[i-1]
    dy = y[i] - y[i-1]
    AUC += ( (dx*y[i-1]) + (0.5*dx*dy) )
  end
  return AUC
end
