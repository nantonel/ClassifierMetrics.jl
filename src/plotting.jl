@recipe function dummy(curve::ROC)
  x, y = get_curve(curve)
	ticks = [0.2, 0.4, 0.6, 0.8, 1]
	xlims := (0,1)
	ylims := (0,1)
	xguide := "False Positive Rate (%)"
	yguide := "True Positive Rate (%)"
	title --> "Receiver Operator Characteristic (ROC)"
	legend --> :outerleft
	ticks --> (ticks, string.(ticks.*100))
	@series begin
		seriescolor --> :black
		linestyle --> :dash
		label := ""
		[0, 1], [0, 1]
	end
  EER, idx = eer(curve)
	@series begin
		seriescolor --> :red
    seriestype := :scatter
		label := ""
    [x[idx+1]], [y[idx+1]]
	end
	@series begin
    x, y 
	end
end

@recipe function dummy(curve::PR)
  x, y = get_curve(curve)
	ticks = [0.2, 0.4, 0.6, 0.8, 1]
	xlims := (0,1)
	ylims := (0,1)
	xguide := "Recall (%)"
	yguide := "Precision (%)"
	title --> "Precision Recall (PR)"
	legend --> :outerleft
	ticks --> (ticks, string.(ticks.*100))
	@series begin # this is just to get the same colors with ROC
		seriescolor --> :black
		linestyle --> :dash
		label := ""
		[0, 0], [0, 0]
	end
  EER, idx = eer(curve)
	@series begin
		seriescolor --> :red
    seriestype := :scatter
		label := ""
    [x[idx+1]], [y[idx+1]]
	end
	@series begin
    x, y
	end
end

@recipe function dummy(curve::DET)
  x, y = get_curve(curve)
	ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60]
	xguide := "False Positive Rate (%)"
	yguide := "False Negative Rate (%)"
	title --> "Detection Error Tradeoff (DET)"
	legend --> :outerleft
	ticks --> (det_axis.( ticks ./ 100), string.(ticks))
	xlims --> (det_axis(0.001), det_axis(0.6))
	ylims --> (det_axis(0.001), det_axis(0.6))
	@series begin # this is just to get the same colors with ROC
		seriescolor --> :black
		linestyle --> :dash
		label := ""
		[0, 0], [0, 0]
	end
  EER, idx = eer(curve)
	@series begin
    seriestype := :scatter
		seriescolor --> :red
		label := ""
    [x[idx]], [y[idx]]
	end
	@series begin
		x, y
	end
end
