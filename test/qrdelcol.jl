Random.seed!(0)

# facts("qrdelcol") do
    m = 100
    A = randn(m,m)
    Q, R = qr(A)
    for i in 100:-1:1
        k = rand(1:i)
        global A = A[:,1:i .!= k]
        global R = qrdelcol(R, k)
        @assert norm( R'*R - A'*A ) < 1e-5
    end
# end
