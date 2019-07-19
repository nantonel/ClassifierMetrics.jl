module ClassifierMetrics

using RecipesBase # for creating a Plots.jl recipe
using SpecialFunctions

include("binarymetrics.jl")
include("roc.jl")
include("det.jl")
include("plotting.jl")

end # module
