module ClassifierMetrics

using RecipesBase # for creating a Plots.jl recipe

include("binarymetrics.jl")
include("roc.jl")
include("plotting.jl")

end # module
