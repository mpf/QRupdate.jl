module QRupdate

using LinearAlgebra

export qraddcol, qraddrow, qrdelcol, csne

"""
Add a column to a QR factorization without using Q.

`R = qraddcol(A,R,v)`  adds the m-vector `v` to the QR factorization of the
m-by-n matrix `A` without using `Q`. On entry, `R` is a dense n-by-n upper
triangular matrix such that `R'*R = A'*A`.

`R = qraddcol(A,R,v,β)` is similar to the above, except that the
routine updates the QR factorization of
```
[A; β I],   and   R'*R = (A'*A + β^2*I).
```

`A` should have fewer columns than rows.  `A` and `v` may be sparse or
dense.  On exit, `R` is the (n+1)-by-(n+1) factor corresponding to

```
  Anew = [A        V ],    Rnew = [R   u  ],   Rnew'Rnew = Anew'Anew.
         [beta*I     ]            [  gamma]
         [       beta]
```

The new column `v` is assumed to be nonzero.
If `A` has no columns yet, input `A = []`, `R = []`.
"""
function qraddcol(A::AbstractMatrix{T}, Rin::AbstractMatrix{T}, a::Vector{T}, β::T = zero(T)) where {T}

    m, n = size(A)
    anorm  = norm(a)
    anorm2 = anorm^2
    β2  = β^2
    if β != 0
        anorm2 = anorm2 + β2
        anorm  = sqrt(anorm2)
    end

    if n == 0
        return reshape([anorm], 1, 1)
    end

    R = UpperTriangular(Rin)

    c      = A'*a           # = [A' β*I 0]*[a; 0; β]
    u      = R'\c
    unorm2 = norm(u)^2
    d2     = anorm2 - unorm2

    if d2 > anorm2 #DISABLE 0.01*anorm2     # Cheap case: γ is not too small
        γ = sqrt(d2)
    else
        z = R\u          # First approximate solution to min ||Az - a||
        r = a - A*z
        c = A'*r
        if β != 0
            c = c - β2*z
        end
        du = R'\c
        dz = R\du
        z  += dz          # Refine z
      # u  = R*z          # Original:     Bjork's version.
        u  += du          # Modification: Refine u
        r  = a - A*z
        γ = norm(r)       # Safe computation (we know gamma >= 0).
        if !iszero(β)
            γ = sqrt(γ^2 + β2*norm(z)^2 + β2)
        end
    end

    # This seems to be faster than concatenation, ie:
    # [ Rin         u
    #   zeros(1,n)  γ ]
    Rout = zeros(T, n+1, n+1)
    Rout[1:n,1:n] .= R
    Rout[1:n,n+1] .= u
    Rout[n+1,n+1] = γ
    # Rout[n+1,1:n] .= 0.0


    return Rout
end

"""
Add a row and update a Q-less QR factorization.

`qraddrow!(R, a)` returns the triangular part of a QR factorization of `[A; a]`, where `A = QR` for some `Q`.  The argument `a` should be a row
vector.
"""
function qraddrow(R::AbstractMatrix{T}, a::AbstractMatrix{T}) where {T}

    n = size(R,1)
    @inbounds @simd for k in 1:n
        G, r = givens( R[k,k], a[k], 1, 2 )
        B = G * [ reshape(R[k,k:n], 1, n-k+1)
                  reshape(a[:,k:n], 1, n-k+1) ]
        R[k,k:n] = B[1,:]
        a[  k:n] = B[2,:]
    end
    return R
end

"""
Delete the k-th column and update a Q-less QR factorization.

`R = qrdelcol(R,k)` deletes the k-th column of the upper-triangular
`R` and restores it to upper-triangular form.  On input, `R` is an n x
n upper-triangular matrix.  On output, `R` is an n-1 x n-1 upper
triangle.

    18 Jun 2007: First version of QRdelcol.m.
                 Michael Friedlander (mpf@cs.ubc.ca) and
                 Michael Saunders (saunders@stanford.edu)
                 To allow for R being sparse,
                 we eliminate the k-th row of the usual
                 Hessenberg matrix rather than its subdiagonals,
                 as in Reid's Bartel-Golub LU update and also
                 the Forrest-Tomlin update.
                 (But Matlab will still be pretty inefficient.)
    18 Jun 2007: R is now the exact size on entry and exit.
    30 Dec 2015: Translate to Julia.
"""
function qrdelcol(R::AbstractMatrix{T}, k::Int) where {T}

    m = size(R,1)
    R = R[:,1:m .!= k]          # Delete the k-th column
    n = size(R,2)               # Should have m=n+1

    for j in k:n                # Forward sweep to reduce k-th row to zeros
        G, y = givens(R[j+1,j], R[k,j], 1, 2)
        R[j+1,j] = y
        if j<n && G.s != 0
            @inbounds @simd for i in j+1:n
                tmp = G.c*R[j+1,i] + G.s*R[k,i]
                R[k,i] = G.c*R[k,i] - conj(G.s)*R[j+1,i]
                R[j+1,i] = tmp
            end
        end
    end
    R = R[1:m .!= k, :]         # Delete the k-th row
    return R
end

"""
    x, r = csne(R, A, b)

solves the least-squares problem

    minimize  ||r||_2,  r := b - A*x

using the corrected semi-normal equation approach described by Bjork (1987). Assumes that `R` is upper triangular.
"""
function csne(Rin::AbstractMatrix{T}, A::AbstractMatrix{T}, b::Vector{T}) where {T}

    R = UpperTriangular(Rin)
    q = A'b
    x = R' \ q

    bnorm2 = sum(abs2, b)
    xnorm2 = sum(abs2, x)
    d2 = bnorm2 - xnorm2

    x = R \ x

    # Apply one step of iterative refinement.
    r = b - A*x
    q = A'r
    dx = R' \ q
    dx = R  \ dx
    x += dx
    r = b - A*x
    return (x, r)
end


end # module
