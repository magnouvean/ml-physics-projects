using LinearAlgebra
using Statistics
using Plots

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: λs, generatedata
using .Functions: showinfo_mse_r2, plotinfo, mse, r2score, calculateridgeintercept

function calculate_mse_r2_ridge(λs, X_train, X_test, y_train, y_test)
    mse_train = zeros(length(λs))
    mse_test = zeros(length(λs))
    r2_train = zeros(length(λs))
    r2_test = zeros(length(λs))
    β_avg_sizes = zeros(length(λs))

    for (i, λ) in enumerate(λs)
        β̂ = pinv(X_train' * X_train + λ * I) * X_train' * y_train
        # We need to re-add the intercept, which we can calculate from the other
        # parameters and the design matrix, see:
        # https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/week36.html
        β̂_0 = calculateridgeintercept(X_train, y_train, β̂)
        ŷ_train = X_train * β̂ .+ β̂_0
        ŷ_test = X_test * β̂ .+ β̂_0

        mse_train[i] = mse(y_train, ŷ_train)
        mse_test[i] = mse(y_test, ŷ_test)
        r2_train[i] = r2score(y_train, ŷ_train)
        r2_test[i] = r2score(y_test, ŷ_test)

        β_avg_sizes[i] = (abs(β̂_0) + sum(abs.(β̂))) / (length(β̂) + 1)
    end

    return mse_train, mse_test, r2_train, r2_test, β_avg_sizes
end

println("Without noise")
X_train, X_test, y_train, y_test = generatedata(5)
mse_train, mse_test, r2_train, r2_test, β_sizes_no_noise = calculate_mse_r2_ridge(λs, X_train, X_test, y_train, y_test)
showinfo_mse_r2(λs, mse_train, mse_test, r2_train, r2_test)
plotinfo(λs, mse_train, mse_test, r2_train, r2_test, "ridge_without_noise", title="Without noise")

println("With noise")
X_train, X_test, y_train, y_test = generatedata(5, add_noise=true)
mse_train, mse_test, r2_train, r2_test, β_sizes_with_noise = calculate_mse_r2_ridge(λs, X_train, X_test, y_train, y_test)
showinfo_mse_r2(λs, mse_train, mse_test, r2_train, r2_test)
plotinfo(λs, mse_train, mse_test, r2_train, r2_test, "ridge_with_noise", title="With noise")

plot([log10.(λs) log10.(λs)],
    [β_sizes_no_noise β_sizes_with_noise],
    label=["without noise" "with noise"],
    xlabel="log10(λ)",
    ylabel="abs(avg(β))")
savefig(dirname(@__DIR__) * "/figures/ridge_beta_size.png")