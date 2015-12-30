srand(0)

facts("qraddrow") do

    m, n = 3, 3
    A = randn(m,m)
    Q, R = qr(A)
    for i in 1:100
        a = randn(m)'
        A = [A; a]
        R = qraddrow(R, a)
        @fact vecnorm( R'R - A'*A ) --> less_than(1e-5)
    end

end
