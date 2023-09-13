module Data

include("./Functions.jl")

using Statistics
using Random: seed!, shuffle
using ScikitLearn.CrossValidation: train_test_split
using Distributions

# Setting the seed for train test splitting and random x1/x2
seed!(1234)

# Our datapoints (these are set randonmly instead of using a range in order for us to get full rank of the matrix)
n = 1000
x1 = rand(n)
x2 = rand(n)

order = 5
# 5 (5th order x1-s) + 5 (5th order x2-s) + 10 (intercations between x1-s and x2-s)
X = zeros((length(x1), 2 * order + (order - 1)^2))
for i in 1:order
    X[:, i] = x1 .^ i
    X[:, order+i] = x2 .^ i
end

for i in 1:(order-1)
    for j in 1:(order-1)
        X[:, 1+order+i*(order-1)+j] = (x1 .^ i) .* (x2 .^ j)
    end
end

y = Functions.frankefunction(x1, x2)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Standard scale
for i in 1:size(X, 2)
    μ̂ = mean(X_train[:, i])
    X_train[:, i] = (X_train[:, i] .- μ̂)
    X_test[:, i] = (X_test[:, i] .- μ̂)
end

# Add response with noise
y_train_with_noise = y_train + rand(Normal(0, 1), length(y_train))
y_test_with_noise = y_test + rand(Normal(0, 1), length(y_test))

end