using ROC
using Random
Random.seed!(123)

## GENERATE SYNTHETIC DATA:
labels = rand(Bool, 200);

# function to add noise
add_noise(label::Bool, λ=0.0) = label ? 1 - λ*rand() : λ*rand()
# simulate good detector 
good = add_noise.(labels, 0.6)
# simulate bad detector 
bad  = add_noise.(labels, 1.0)


roc_good = roc(good, labels, true);
roc_bad = roc(bad, labels, true);
println("AUC (good detector): $(AUC(roc_good)) ")
println("AUC (bad  detector): $(AUC(roc_bad )) ")

using Plots
gr()

p = plot(roc_good, label="good");
plot!(p,roc_bad, label="bad")

savefig("rocs.png")
