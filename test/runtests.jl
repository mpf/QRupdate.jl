using QRupdate
import QRupdate: solveR!, solveRT!
using Test
using LinearAlgebra

MAX_SIZE = 20

@testset "qraddcol!" begin
    m = MAX_SIZE
    T = ComplexF64
    work = rand(T, m)
    work2 = rand(T, m)
    work3 = rand(T, m)
    work4 = rand(T, m)
    work5 = rand(T, m)
    A = randn(T, m, 0)
    R = Array{T, 2}(undef, 0, 0)
    Rin = zeros(T,m,m)
    Ain = zeros(T,m,m)
    for i in 1:m
        a = randn(T, m)
        R = qraddcol(A, R, a, zero(T))
        A = [A a]
        Q = A*inv(R)
        qraddcol!(Ain, Rin, a, i-1, work, work2, work3, work4, work5)
        Qin = view(Ain, :, 1:i)*inv(view(Rin, 1:i, 1:i))
        @test norm(Q'Q - diagm(ones(size(Q,2)))) < 1e-10
        @test norm(Qin'Qin - diagm(ones(size(Qin,2)))) < 1e-10
    end
end



@testset "csne!" begin
    
    # work vecs
    m = MAX_SIZE
    T = ComplexF64
    work = rand(T, m)
    work2 = rand(T, m)
    work3 = rand(T, m)
    work4 = rand(T, m)
    work5 = rand(T, m)
    sol = zeros(T, m)

    A = randn(T, m, 0)
    R = Array{T, 2}(undef, 0, 0)
    Rin = zeros(T,m,m)
    Ain = zeros(T,m,m)

    b = randn(T, m)
    for i in 1:m
        a = randn(T, m)
        R = qraddcol(A, R, a)
        A = [A a]
        qraddcol!(Ain, Rin, a, i-1, work, work2, work3, work4, work5)
        x, r = csne(R, A, b)
        csne!(Rin, Ain, b, sol, work, work2, work3, work4, i)
        @test norm(x - view(sol, 1:i)) < 1e-8
    end
end


@testset "qraddcol with β = 0" begin
    m = MAX_SIZE
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
    m = MAX_SIZE
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
    m = MAX_SIZE
    A = randn(m,m)
    Q, R = qr(A)
    for i in MAX_SIZE:-1:1
        k = rand(1:i)
        A = A[:,1:i .!= k]
        R = qrdelcol(R, k)
        @test norm( R'*R - A'*A ) < 1e-5
    end
end

@testset "qrdelcol!" begin
    m = MAX_SIZE
    A = randn(m,m)
    Ain = copy(A)
    Q, R = qr(A)
    Qin = deepcopy(Q)
    Rin = deepcopy(R)
    actual_size = m
    for i in MAX_SIZE:-1:1
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
    for m in 1:MAX_SIZE
        A = qr(rand(m,m)).R
        b = ones(m)
        rhs = A * b
        sol = zeros(m)
        solveR!(A,rhs,sol)
        @test norm(A * sol - rhs) < 1e-10
    end
end

@testset "solveRT!" begin
    for m in 1:MAX_SIZE
        A = qr(rand(m,m)).R
        b = ones(m)
        rhs = A' * b
        sol = zeros(m)
        solveRT!(A,rhs,sol)
        @test norm(A' * sol - rhs) < 1e-10
    end
end



