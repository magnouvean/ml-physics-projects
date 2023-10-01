using Plots

include("./Functions.jl")
using .Functions: frankefunction

n = 100
x = range(0, 1, length=n)
y = range(0, 1, length=n)

surface(x,
    y,
    frankefunction,
    title="frankefunction for values in [0, 1] and n=$(n)",
    xlabel="x",
    ylabel="y",
    zlabel="z")
savefig(dirname(@__DIR__) * "/figures/frankefunction.png")