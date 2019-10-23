Random.seed!(0)

# facts("qraddcol") do

    # context("β = 0") do
        m = 100
        A = randn(m,0)
        R = Array{Float64}(undef, 0,0)
        r = Vector{Float64}(undef, m)
        
        for i in 1:m
            a = randn(m)
            Rout = Array{Float64}(undef, i, i)
            qraddcol!(A, R, a, Rout, r)
            global R = qraddcol(A, R, a)
            global A = [A a]
            @assert  norm( R'*R - A'*A ) < 1e-5
            @assert  norm( Rout'*Rout - A'*A ) < 1e-5 "$(norm( Rout'*Rout - A'*A )) is not less than 1e-5"
        end

        b = sum(A, dims=2)[:]
        (x, rtest) = csne(R, A, b)
        @assert norm(rtest) < 1e-5
        @assert norm(x .- 1) < 1e-5

        x = csne!(R, A, b, r)
        @assert norm(r) < 1e-5
        @assert norm(x .- 1) < 1e-5

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
