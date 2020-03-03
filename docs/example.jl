using ClassifierMetrics 
using Random
Random.seed!(123)

# Generate synthetic data
labels = rand(Bool, 500);

# function to add noise
add_noise(label::Bool, λ=0.0) = label ? 1 - λ*rand() : λ*rand()
# simulate good classifier 
good = add_noise.(labels, 0.6)
# simulate bad classifier
bad  = add_noise.(labels, 0.98)

roc_good = ROC(labels, good)
roc_bad  = ROC(labels,  bad)

pr_bad  = PR(labels,  bad)
pr_good  = PR(labels,  good)

det_bad  = DET(labels,  bad)
det_good  = DET(labels,  good)

println("ROCAUC (good detector): $(auc(roc_good)) ")
println("ROCAUC (bad  detector): $(auc(roc_bad )) ")

println("PRAUC (good detector): $(auc(pr_good)) ")
println("PRAUC (bad  detector): $(auc(pr_bad )) ")

using Plots
gr()

roc_plot = plot(roc_good, label="good");
plot!(roc_plot,roc_bad, label="bad", legend=:bottomright)

pr_plot = plot(pr_good);
plot!(pr_plot,pr_bad, legend=false)

det_plot = plot(det_good);
plot!(det_plot,det_bad, legend=false)

plot(roc_plot, pr_plot, det_plot)
