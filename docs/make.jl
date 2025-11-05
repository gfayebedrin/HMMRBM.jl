using Documenter
using Documenter.Remotes: GitHub
using HMMRBM
using DensityInterface
using HiddenMarkovModels

makedocs(
    sitename="HMMRBM.jl",
    authors="Guillaume Faye-Bedrin",
    repo=GitHub("gfayebedrin", "HMMRBM.jl"),
    pages=[
        "Home" => "index.md",
        "Multi-sequence HMMs" => "HMM.md",
        "RBM emissions" => "RBMdist.md",
        "Internal utilities" => "internal.md",
    ],
)

deploydocs(
    repo="github.com/gfayebedrin/HMMRBM.jl.git",
    devbranch="main",
    versions=[
        "stable" => "v^", # latest tagged release
        "dev" => "main",
    ]
)
