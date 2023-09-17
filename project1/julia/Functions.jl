module Functions

using Statistics
using Plots

export frankefunction

function frankefunction(x, y)
    term1 = 0.75 * exp.(-0.25 * (9 * x .- 2) .^ 2 - 0.25 * (9 * y .- 2) .^ 2)
    term2 = 0.75 * exp.(-((9 * x .+ 1) .^ 2) / 49.0 - 0.1 * (9 * y .+ 1))
    term3 = 0.5 * exp.(-((9 * x .- 7) .^ 2) / 4.0 - 0.25 * (9 * y .- 3) .^ 2)
    term4 = -0.2 * exp.(-(9 * x .- 4) .^ 2 - (9 * y .- 7) .^ 2)
    return term1 + term2 + term3 + term4
end

mse(y, ŷ) = mean((y - ŷ) .^ 2)
r2score(y, ŷ) = 1 - (mean((y - ŷ) .^ 2)) / (mean((y .- mean(y)) .^ 2))

# Prints the MSE and R^2 metric values given some different mses for different λ
# values typically obtained using ridge/lasso.
function showinfo(λs, mse_train, mse_test, r2_train, r2_test)
    println("Best λ (mse, train): $(λs[argmin(mse_train)]), mse: $(minimum(mse_train))")
    println("Best λ (mse, test): $(λs[argmin(mse_test)]), mse: $(minimum(mse_test))")
    println("Best λ (R^2, train): $(λs[argmax(r2_train)]), R^2: $(maximum(r2_train))")
    println("Best λ (R^2, test): $(λs[argmax(r2_test)]), R^2: $(maximum(r2_test))")
end

function plotinfo(λs, mse_train, mse_test, r2_train, r2_test)
    plot([log10.(λs), log10.(λs)], [mse_train, mse_test], label=["train" "test"])
    plot([log10.(λs), log10.(λs)], [r2_train, r2_test], label=["train" "test"])
end

end