using QRupdate
using LinearAlgebra
using BenchmarkTools

for mm in [1000,2000,4000,10000, 20000,50000,100000]
    #reset_timer()
    m, n = mm, 100
    A = randn(m, n)
    R = qr(A).R
    Rin = deepcopy(R)
    Ain = deepcopy(A)

    actual_size = n
    i = 20
    println("====== R ", i)
    @btime $R = qrdelcol($R, $i)
    println("====== Rin ", i)
    @btime qrdelcol!($Ain, $Rin, $i)
end
