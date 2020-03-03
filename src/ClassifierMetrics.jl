module ClassifierMetrics

using RecipesBase # for creating a Plots.jl recipe
using SpecialFunctions

include("utils.jl")
include("binarymetrics.jl")
include("roc.jl")
include("det.jl")
include("pr.jl")
include("plotting.jl")

end # module
