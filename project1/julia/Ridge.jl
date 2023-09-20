using LinearAlgebra
using Statistics

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: λs, generatedata
using .Functions: showinfo, plotinfo, mse, r2score

function calculate_mse_r2_ridge(λs, X_train, X_test, y_train, y_test)
    mse_train = zeros(length(λs))
    mse_test = zeros(length(λs))
    r2_train = zeros(length(λs))
    r2_test = zeros(length(λs))

    for (i, λ) in enumerate(λs)
        β̂ = pinv(X_train' * X_train + λ * I) * X_train' * y_train
        # We need to re-add the intercept, which we can calculate from the other
        # parameters and the design matrix, see:
        # https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/week36.html
        n = size(X_train, 1)
        p = size(X_train, 2)
        β̂_0 = mean(y_train) - 1 / n * sum([sum([X_train[i, j] * β̂[j] for j in 1:p]) for i in 1:n])
        ŷ_train = X_train * β̂ .+ β̂_0
        ŷ_test = X_test * β̂ .+ β̂_0

        mse_train[i] = mse(y_train, ŷ_train)
        mse_test[i] = mse(y_test, ŷ_test)
        r2_train[i] = r2score(y_train, ŷ_train)
        r2_test[i] = r2score(y_test, ŷ_test)

    end

    return mse_train, mse_test, r2_train, r2_test
end

# Without noise
X_train, X_test, y_train, y_test = generatedata(5)
mse_train, mse_test, r2_train, r2_test = calculate_mse_r2_ridge(λs, X_train, X_test, y_train, y_test)
showinfo(λs, mse_train, mse_test, r2_train, r2_test)
plotinfo(λs, mse_train, mse_test, r2_train, r2_test)

# With noise
X_train, X_test, y_train, y_test = generatedata(5, true, false, true)
mse_train, mse_test, r2_train, r2_test = calculate_mse_r2_ridge(λs, X_train, X_test, y_train, y_test)
showinfo(λs, mse_train, mse_test, r2_train, r2_test)
plotinfo(λs, mse_train, mse_test, r2_train, r2_test)