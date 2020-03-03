@recipe function dummy(curve::ROC)
	ticks = [0.2, 0.4, 0.6, 0.8, 1]
	xlim := (0,1)
	ylim := (0,1)
	xlab := "False Positive Rate (%)"
	ylab := "True Positive Rate (%)"
	title --> "Receiver Operator Characteristic (ROC)"
	legend --> :outerleft
	ticks --> (ticks, string.(ticks.*100))
	@series begin
		color --> :black
		linestyle --> :dash
		label := ""
		[0, 1], [0, 1]
	end
  EER, idx = eer(curve)
	@series begin
		color --> :red
    seriestype := :scatter
		label := ""
    [curve.FPR[idx]], [curve.TPR[idx]]
	end
	@series begin
    [curve.FPR[1];curve.FPR], [0;curve.TPR]
	end
end

@recipe function dummy(curve::PR)
	ticks = [0.2, 0.4, 0.6, 0.8, 1]
	xlim := (0,1)
	ylim := (0,1)
	xlab := "Recall (%)"
	ylab := "Precision (%)"
	title --> "Precision Recall (PR)"
	legend --> :outerleft
	ticks --> (ticks, string.(ticks.*100))
	@series begin # this is just to get the same colors with ROC
		color --> :black
		linestyle --> :dash
		label := ""
		[0, 0], [0, 0]
	end
  EER, idx = eer(curve)
	@series begin
		color --> :red
    seriestype := :scatter
		label := ""
    [curve.recall[idx]], [curve.precision[idx]]
	end
	@series begin
    curve.recall, curve.precision
	end
end

@recipe function dummy(curve::DET)
	f = x -> sqrt(2) * erfinv(2*x-1)
	ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60]
	xlab := "False Positive Rate (%)"
	ylab := "False Negative Rate (%)"
	title --> "Detection Error Tradeoff (DET)"
	legend --> :outerleft
	ticks --> (f.( ticks ./ 100), string.(ticks))
	xlim --> (f(0.001), f(0.6))
	ylim --> (f(0.001), f(0.6))
	@series begin # this is just to get the same colors with ROC
		color --> :black
		linestyle --> :dash
		label := ""
		[0, 0], [0, 0]
	end
  EER, idx = eer(curve)
	@series begin
    seriestype := :scatter
		color --> :red
		label := ""
    [f.(curve.FPR[idx])], [f.(curve.FNR[idx])]
	end
	@series begin
		f.(curve.FPR), f.(curve.FNR)
	end
end
