using Documenter
using HMMRBM

makedocs(
    sitename="HMMRBM.jl",
    # modules=[HMMRBM],
    pages=[
        "Home" => "index.md",
        "Multi-sequence HMMs" => "HMM.md",
        "RBM emissions" => "RBMdist.md",
    ],
)
