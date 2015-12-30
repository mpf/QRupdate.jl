# QRupdate

[![Build Status](https://travis-ci.org/mpf/QRupdate.jl.svg?branch=master)](https://travis-ci.org/mpf/QRupdate.jl)
[![codecov.io](https://codecov.io/github/mpf/QRupdate.jl/coverage.svg?branch=master)](https://codecov.io/github/mpf/QRupdate.jl?branch=master)

Update the "Q-less" QR factorization of a matrix. Routines are
provided for adding and deleting columns, adding rows, and solving the
least-squares systems.

## Installing

```JULIA
Pkg.clone("https://github.com/mpf/QRupdate.jl.git")
Pkg.test("QRupdate")
```

## Examples

### Adding columns
Build the "Q-less" QR factorization of A one column at a time.
```JULIA
m, n = 100, 50
A = randn(m,0)
R = Array{Float64}(0,0)
for i in 1:n
    a = randn(m)
    R = qraddcol(A, R, a)
    A = [A a]
end
```
Solve a least-squares problem using R.
```JULIA
b = randn(m)
x, r = csne(R, A, b)
```

### Deleting columns
Delete a column and compute new R.
```JULIA
n = size(A,2)
k = rand(1:n)
A = A[:, 1:n .!= k]
R = qrdelcol(R, k)
```
Solve a least-squares problem using R.
```JULIA
x, r = csne(R, A, b)
```

### Adding rows
Add a row to A.
```JULIA
n = size(A,2)
a = randn(n)'  # must be row vector
A = [A; a]
R = qraddrow(R, a)
```
Solve a least-squares problem using R.
```JULIA
b = [b; randn()]
x, r = csne(R, A, b)
```

## Reference
Björck, A. (1996). Numerical methods for least squares problems. SIAM.
