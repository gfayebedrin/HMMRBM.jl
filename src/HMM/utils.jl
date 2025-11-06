# utils.jl

# Utility functions for HMMs

sum_to_one!(x) = ldiv!(sum(x), x)

"""
    gradient_descent(value_gradient_hessian, x0;
                     lr        = 1e-2,
                     maxiter   = 1_000,
                     tol       = 1e-6,
                     use_hess  = true)

Plain (optionally Hessian-pre-conditioned) gradient descent. The function `value_gradient_hessian`
must return a tuple `(f, g, H)` containing the objective value, gradient, and
Hessian evaluated at the supplied point. The Hessian can be `nothing`.

Returns a named tuple:
    (xmin = ...,      # minimiser
     fmin = ...,      # final objective value
     converged = ..., # Bool
     niter = ...)     # iterations actually executed
"""
function gradient_descent(value_gradient_hessian, x0;
    lr=1e-1,
    maxiter=100,
    tol=1e-3,
)

    x = copy(x0)
    g = similar(x)
    H = Diagonal(similar(x))
    fx, g, H = value_gradient_hessian(x)


    for k in 1:maxiter
        if norm(g) ≤ tol
            return (; xmin=x, fmin=fx, converged=true, niter=k)
        end

        d = if H !== nothing
            diagH = H isa AbstractVector ? H : diag(H)
            g ./ (diagH .+ 1e-8)
        else
            g
        end
        x_new = x .- lr .* d
        fx_new, g_new, H_new = value_gradient_hessian(x_new)

        if fx_new < fx          # accept
            x, fx, g, H = x_new, fx_new, g_new, H_new
        else                    # reject & shrink step
            lr *= 0.5
        end
    end

    return (; xmin=x, fmin=fx, converged=false, niter=maxiter)
end

function gradient_ascent(value_gradient_hessian, x0;
    lr=1e-2,
    maxiter=1_000,
    tol=1e-6,
    )

    (; xmin, fmin, converged, niter) = gradient_descent(x -> begin
            fx, g, H = value_gradient_hessian(x)
            (-fx, -g, H === nothing ? nothing : -H)
        end, x0; lr, maxiter, tol)

    return (; xmax=xmin, fmax=-fmin, converged, niter)
end
