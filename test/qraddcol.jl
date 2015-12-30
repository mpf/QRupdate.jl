srand(0)

facts("qraddcol") do

    context("β = 0") do
        m = 100
        A = randn(m,0)
        R = Array{Float64}(0,0)
        
        for i in 1:m
            a = randn(m)
            R = qraddcol(A, R, a)
            A = [A a]
            @fact  vecnorm( R'*R - A'*A ) --> less_than(1e-5)
        end
    end

    context("β > 0") do
        m = 100
        A = randn(m,0)
        R = Array{Float64}(0,0)
        β = 0.1
        for i in 1:m
            a = randn(m)
            B = [A a]
            R = qraddcol(A, R, a, β)
            @fact vecnorm( R'*R - B'*B - β^2*I ) --> less_than(1e-5)
        end
    end

end
