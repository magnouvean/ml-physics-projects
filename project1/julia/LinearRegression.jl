using LinearAlgebra
using Statistics
using Printf: @printf

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score

function linearregression(X_train, X_test, y_train, y_test)
    β̂ = pinv(X_train' * X_train) * X_train' * y_train
    ŷ_train = X_train * β̂
    ŷ_test = X_test * β̂
    @printf "MSE train: %.6f, MSE test: %.6f\n" mse(y_train, ŷ_train) mse(y_test, ŷ_test)
    @printf "R^2 train: %.6f, R^2 test: %.6f\n" r2score(y_train, ŷ_train) r2score(y_test, ŷ_test)
end

println("Without noise")
X_train, X_test, y_train, y_test = generatedata(5, include_intercept=true)
linearregression(X_train, X_test, y_train, y_test)

println("With noise")
X_train, X_test, y_train, y_test = generatedata(5, include_intercept=true, add_noise=true)
linearregression(X_train, X_test, y_train, y_test)