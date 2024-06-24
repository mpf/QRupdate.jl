using QRupdate
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
    Q, R = qr(A)
    R2 = qrdelcol(R, 10)
    qrdelcol!(R, 10)
    @test norm(R) - norm(R2) < 1e-14
end


@testset "qraddcol!" begin
    m = 100
    A = randn(m, 0)
    b = randn(m)
    R = zeros(0, 0)
    
    # Do two steps
    a1 = randn(m)
    R = qraddcol(A, R, a1)
    A = [A a1]
    a2 = randn(m)
    R = qraddcol(A, R, a2)
    A = [A a2]

    # Now compute two steps of inplace
    Rin = zeros(2,2)
    Ain = zeros(m, 2)
    qraddcol!(Ain, Rin, a1, 0)
    qraddcol!(Ain, Rin, a2, 1)
    @test norm(R) - norm(Rin) < 1e-10
end

