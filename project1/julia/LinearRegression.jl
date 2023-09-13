using LinearAlgebra
using Statistics

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: X_train, X_test, y_train, y_test
using .Functions: mse, r2score

# We add 1-s to our first columns back in order to get an intercept (this was
# removed for scaling purposes).
X_train_new = [ones(size(X_train, 1)) X_train]
X_test_new = [ones(size(X_test, 1)) X_test]

β̂ = inv(X_train_new' * X_train_new) * X_train_new' * y_train
ŷ_train = X_train_new * β̂
ŷ_test = X_test_new * β̂
println(mse(y_train, ŷ_train))
println(mse(y_test, ŷ_test))
println(r2score(y_train, ŷ_train))
println(r2score(y_test, ŷ_test))