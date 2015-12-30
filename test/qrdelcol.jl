srand(0)

facts("qrdelcol") do
    m = 100
    A = randn(m,m)
    Q, R = qr(A)
    for i in 100:-1:1
        k = rand(1:i)
        A = A[:,1:i .!= k]
        R = qrdelcol(R, k)
        @fact vecnorm( R'*R - A'*A ) --> less_than(1e-5)
    end
end
