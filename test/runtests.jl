using QRupdate
using Test
using LinearAlgebra

tests = ["qraddcol",
         "qraddrow",
         "qrdelcol"]

for t in tests
    include("$(t).jl")
end
