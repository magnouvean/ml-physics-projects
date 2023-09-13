using LinearAlgebra

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: X_train, X_test, y_train, y_test
using .Functions: mse, r2score

λs = range(0.01, 5.0, length=50)

mse_train = Dict()
mse_test = Dict()
for λ in λs
    β̂ = inv(X_train' * X_train + λ * I) * X_train' * y_train
    ŷ_train = X_train * β̂
    ŷ_test = X_test * β̂
    mse_train[λ] = mse(y_train, ŷ_train)
    mse_test[λ] = mse(y_test, ŷ_test)
end

using Plots
scatter([collect(keys(mse_train)), collect(keys(mse_test))],
    [collect(values(mse_train)), collect(values(mse_test))],
    label=["train", "test"])