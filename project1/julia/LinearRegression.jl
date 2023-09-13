using LinearAlgebra

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: X_train, X_test, y_train, y_test
using .Functions: mse, r2score

β̂ = inv(X_train' * X_train) * X_train' * y_train
ŷ_train = X_train * β̂
ŷ_test = X_test * β̂
println(mse(y_train, ŷ_train))
println(mse(y_test, ŷ_test))
println(r2score(y_train, ŷ_train))
println(r2score(y_test, ŷ_test))