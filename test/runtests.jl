using QRupdate
import QRupdate: solveR!, solveRT!
using Test
using LinearAlgebra

@testset "qraddcol with β = 0" begin
    m = 100
    A = randn(m, 0)
    b = randn(m)
    R = zeros(0, 0)
    
    for i in 1:m
        a = randn(m)
        R = qraddcol(A, R, a)
        A = [A a]
        x, r = csne(R, A, b)
        @test norm(A'r, Inf) < 1e-5 
        @test norm( R'*R - A'*A ) < 1e-5
    end
end

@testset "qraddcol with β > 0" begin
    m = 100
    A = randn(m, 0)
    b = randn(m)
    R = zeros(0, 0)
    β = 0.1
    for i in 1:m
        a = randn(m)
        R = qraddcol(A, R, a, β)
        A = [A a]
        # x, r = csne(R, A, b) # TODO: doesn't yet work with β > 0
        # @test norm(A'r, Inf) < 1e-5 
        @test norm( R'*R - A'*A - β^2*I ) < 1e-5
    end
end

@testset "qraddrow" begin
    m, n = 3, 3
    A = randn(m,m)
    Q, R = qr(A)
    for i in 1:100
        a = randn(m)'
        A = [A; a]
        R = qraddrow(R, a)
        @test norm( R'R - A'*A ) < 1e-5
    end
    
end

@testset "qrdelcol" begin
    m = 100
    A = randn(m,m)
    Q, R = qr(A)
    for i in 100:-1:1
        k = rand(1:i)
        A = A[:,1:i .!= k]
        R = qrdelcol(R, k)
        @test norm( R'*R - A'*A ) < 1e-5
    end
end

@testset "qrdelcol!" begin
    m = 100
    A = randn(m,m)
    Ain = copy(A)
    Q, R = qr(A)
    Qin = deepcopy(Q)
    Rin = deepcopy(R)
    actual_size = m
    for i in 100:-1:1
        k = rand(1:i)
        A = A[:,1:i .!= k]
        R = qrdelcol(R, k)
        qrdelcol!(Ain, Rin, k)
        actual_size -= 1
        @test norm(R - Rin[1:actual_size,1:actual_size]) < 1e-10
        @test norm(A - Ain[:,1:actual_size]) < 1e-10
    end
end

@testset "solveR!" begin
    for m in 1:100
        A = qr(rand(m,m)).R
        b = ones(m)
        rhs = A * b
        sol = zeros(m)
        solveR!(A,rhs,sol,m)
        @test norm(A * sol - rhs) < 1e-10
    end
end

@testset "solveRT!" begin
    for m in 1:100
        A = qr(rand(m,m)).R
        b = ones(m)
        rhs = A' * b
        sol = zeros(m)
        solveRT!(A,rhs,sol,m)
        @test norm(A' * sol - rhs) < 1e-10
    end

end

@testset "qraddcol!" begin
    m = 100
    A = randn(m, 0)
    R = Array{Float64, 2}(undef, 0, 0)
    Rin = zeros(m,m)
    Ain = zeros(m, m)
    actual_size = 0
    for i in 1:m
        a = randn(m)
        R = qraddcol(A, R, a)
        A = [A a]
        qraddcol!(Ain, Rin, a, i-1)
        actual_size += 1
        @test norm(R - Rin[1:actual_size,1:actual_size]) < 1e-10
        @test norm(A - Ain[:,1:actual_size]) < 1e-10
    end
end


