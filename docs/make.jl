using Documenter
using HMMRBM

makedocs(
    sitename="HMMRBM.jl",
    authors="Guillaume Faye-Bedrin",
    repo="https://github.com/gfayebedrin/HMMRBM.jl",
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
