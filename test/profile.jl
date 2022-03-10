using QRupdate
using BenchmarkTools

function profile_csne()
	m, n = 100, 50
	b = randn(m)
	A = randn(m, 0)
	R = Array{Float64, 2}(undef, 0, 0)
	for i in 1:n
	    a = randn(m)
	    R = qraddcol(A, R, a)
		A = [A a]
		csne(R, A, b)
	end
	nothing
end

function profile_ls()
	m, n = 100, 50
	b = randn(m)
	A = randn(m, 0)
	for i in 1:n
	    a = randn(m)
		A = [A a]
		A \ b
	end
	nothing
end
@benchmark profile_csne()
@benchmark profile_ls()

