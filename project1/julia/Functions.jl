module Functions

using Statistics

export frankefunction

function frankefunction(x, y)
    term1 = 0.75 * exp.(-0.25 * (9 * x .- 2) .^ 2 - 0.25 * (9 * y .- 2) .^ 2)
    term2 = 0.75 * exp.(-((9 * x .+ 1) .^ 2) / 49.0 - 0.1 * (9 * y .+ 1))
    term3 = 0.5 * exp.(-((9 * x .- 7) .^ 2) / 4.0 - 0.25 * (9 * y .- 3) .^ 2)
    term4 = -0.2 * exp.(-(9 * x .- 4) .^ 2 - (9 * y .- 7) .^ 2)
    return term1 + term2 + term3 + term4
end

mse(y, ŷ) = mean((y - ŷ) .^ 2)
r2score(y, ŷ) = 1 - (mean((y - ŷ) .^ 2)) / (mean((y - mean(y)) .^ 2))

end