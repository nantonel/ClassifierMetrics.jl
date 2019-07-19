module ClassifierMetrics

using Infinity
using Missings # to use Missings.T
using RecipesBase # for creating a Plots.jl recipe

export	BinaryClassifierData, roc, AUC, PPV, cutoffs

include("BinClassEval.jl")
include("rocs.jl")
#include("roc_main.jl")
#include("rocplot.jl")

end # module
