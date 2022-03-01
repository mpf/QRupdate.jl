# QRupdate

![Test Status](https://github.com/mpf/QRupdate.jl/actions/workflows/Test.yml/badge.svg)

Update the "Q-less" QR factorization of a matrix. Routines are
provided for adding and deleting columns, adding rows, and solving
associated linear least-squares problems.

## Installing

```julia
Pkg.add(url="https://github.com/mpf/QRupdate.jl")
```

## Examples

### Adding columns

Build the "Q-less" QR factorization of A one column at a time.

```julia
m, n = 100, 50
A = randn(m,0)
R = Array{Float64, 2}(undef, 0, 0)
for i in 1:n
    a = randn(m)
    R = qraddcol(A, R, a)
    A = [A a]
end
```

Solve a least-squares problem using R.

```julia
b = randn(m)
x, r = csne(R, A, b)
```

### Deleting columns

Delete a column and compute new R.

```julia
n = size(A,2)
k = rand(1:n)
A = A[:, 1:n .!= k]
R = qrdelcol(R, k)
```

Solve a least-squares problem using R.

```julia
x, r = csne(R, A, b)
```

### Adding rows

Add a row to A.

```julia
n = size(A,2)
a = randn(n)'  # must be row vector
A = [A; a]
R = qraddrow(R, a)
```

Solve a least-squares problem using R.

```julia
b = [b; randn()]
x, r = csne(R, A, b)
```

## Reference

Björck, A. (1996). Numerical methods for least squares problems. SIAM.

## Change Log

- 15 Jun 2007: First version of QRaddcol.m (without β).
    - Where necessary, Ake Bjorck's CSNE (Corrected Semi-Normal Equations) method is used to improve the accuracy of `u` and `γ`.  See p143 of Ake Bjork's Least Squares book.

- 18 Jun 2007: `R` is now the exact size on entry and exit.
- 19 Oct 2007: Sparse `A`, a makes `c` and `u` sparse. Force them to be dense.
- 04 Aug 2008: Update `u` using `du`, rather than `u = R*z` as in Ake's book.  We guess that it might be slightly more accurate, but it's hard to tell.  No `R*z` makes it a little cheaper.
- 03 Sep 2008: Generalize `A` to be `[A; β*I]` for some scalar `β`.  Update `u` using `du`, but keep Ake's version in comments.
- 29 Dec 2015: Converted to Julia.
