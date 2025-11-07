# utils.jl

# Utility functions for HMMs

sum_to_one!(x) = ldiv!(sum(x), x)
