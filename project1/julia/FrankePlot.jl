using Plots

include("./Functions.jl")
using .Functions: frankefunction

n = 100
x = range(0, 1, length=n)
y = range(0, 1, length=n)

surface(x,
    y,
    frankefunction,
    xlims=(-0.1, 1.1),
    ylims=(-0.1, 1.1))
savefig(dirname(@__DIR__) * "/figures/frankefunction.png")