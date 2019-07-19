using ClassifierMetrics 
using Random
Random.seed!(123)

# Generate synthetic data
labels = rand(Bool, 200);

# function to add noise
add_noise(label::Bool, λ=0.0) = label ? 1 - λ*rand() : λ*rand()
# simulate good classifier 
good = add_noise.(labels, 0.6)
# simulate bad classifier
bad  = add_noise.(labels, 1.0)


roc_good = ROC(labels, good)
roc_bad  = ROC(labels,  bad)

println("AUC (good detector): $(auc(roc_good)) ")
println("AUC (bad  detector): $(auc(roc_bad )) ")

using Plots
gr()

p = plot(roc_good, label="good");
plot!(p,roc_bad, label="bad")

#savefig("rocs.png")
