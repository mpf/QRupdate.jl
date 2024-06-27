using QRupdate
import QRupdate: solveR!, solveRT!
using Test
using LinearAlgebra

@testset "qraddcol!" begin
    m = 100
    work = rand(m)
    work2 = rand(m)
    work3 = rand(m)
    work4 = rand(m)
    work5 = rand(m)
    A = randn(m, 0)
    R = Array{Float64, 2}(undef, 0, 0)
    Rin = zeros(m,m)
    Ain = zeros(m, m)
    for i in 1:m
        a = randn(m)
        R = qraddcol(A, R, a)
        A = [A a]
        qraddcol!(Ain, Rin, a, i-1, work, work2, work3, work4, work5)
        @test norm(R) - norm(Rin) < 1e-10
        @test norm(A) - norm(Ain) < 1e-10
    end
end



@testset "csne!" begin
    
    # work vecs
    m = 100
    work = rand(m)
    work2 = rand(m)
    work3 = rand(m)
    work4 = rand(m)
    work5 = rand(m)
    sol = zeros(m)
    A = randn(m, 0)
    R = Array{Float64, 2}(undef, 0, 0)
    b = randn(m)
    Rin = zeros(m,m)
    Ain = zeros(m, m)
    for i in 1:m
        a = randn(m)
        R = qraddcol(A, R, a)
        A = [A a]
        qraddcol!(Ain, Rin, a, i-1, work, work2, work3, work4, work5)
        x, r = csne(R, A, b)
        csne!(Rin, Ain, b, sol, work, work2, work3, work4, i)
        @test norm(x) - norm(sol) < 1e-8
    end
end


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
    Qin, Rin = qr(A)
    for i in 100:-1:1
        k = rand(1:i)
        A = A[:,1:i .!= k]
        R = qrdelcol(R, k)
        qrdelcol!(Ain, Rin, k)
        @test norm(R) - norm(Rin) < 1e-10
        @test norm(A) - norm(Ain) < 1e-10
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



