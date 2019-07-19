@recipe function dummy(curve::ROC)
    xlim := (0,1)
    ylim := (0,1)
    xlab := "false positive rate"
    ylab := "true positive rate"
    title --> "Receiver Operator Characteristic"
    legend --> :bottomright
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
