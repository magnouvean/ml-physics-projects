using LinearAlgebra
using Statistics

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score

function linearregression(X_train, X_test, y_train, y_test)
    β̂ = inv(X_train' * X_train) * X_train' * y_train
    ŷ_train = X_train * β̂
    ŷ_test = X_test * β̂
    println(mse(y_train, ŷ_train))
    println(mse(y_test, ŷ_test))
    println(r2score(y_train, ŷ_train))
    println(r2score(y_test, ŷ_test))
end

println("Without noise")
X_train, X_test, y_train, y_test = generatedata(5, true, true)
linearregression(X_train, X_test, y_train, y_test)

println("With noise")
X_train, X_test, y_train, y_test = generatedata(5, true, true, true)
linearregression(X_train, X_test, y_train, y_test)