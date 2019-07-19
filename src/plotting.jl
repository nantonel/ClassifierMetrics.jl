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
	@series begin
		curve.FPR, curve.TPR
	end
end

@recipe function dummy(curve::DET)
	f = x -> sqrt(2) * erfinv(2*x-1)
	ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 40, 60]
	xlab := "False Negative Rate (%)"
	ylab := "False Positive Rate (%)"
	title --> "Detection Error Tradeoff (DET)"
	legend --> :outerleft
	ticks --> (f.( ticks ./ 100), string.(ticks))
	xlim --> (f(0.001), f(0.6))
	ylim --> (f(0.001), f(0.6))
	@series begin
		f.(curve.FPR), f.(curve.FNR)
	end
end
