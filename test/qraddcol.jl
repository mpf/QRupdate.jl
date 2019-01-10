Random.seed!(0)

# facts("qraddcol") do

    # context("β = 0") do
        m = 100
        A = randn(m,0)
        R = Array{Float64}(undef, 0,0)
        
        for i in 1:m
            a = randn(m)
            global R = qraddcol(A, R, a)
            global A = [A a]
            @assert  norm( R'*R - A'*A ) < 1e-5
        end
    # end

    # context("β > 0") do
        m = 100
        A = randn(m,0)
        R = Array{Float64}(undef, 0,0)
        β = 0.1
        for i in 1:m
            a = randn(m)
            B = [A a]
            global R = qraddcol(A, R, a, β)
            @assert norm( R'*R - B'*B - β^2*I ) < 1e-5
        end
    # end

# end
