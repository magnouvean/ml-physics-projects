module Data

include("./Functions.jl")

using Statistics
using Random: seed!, shuffle
using Distributions

# Some ridge/lasso type penalties commonly used by this projects
λs = 10.0 .^ range(-12, 2, length=101)

# Does a simple form on standard scaling by subtracting the mean of each column
# from each item in the column (i.e. it ensures all the columns will have mean
# 0). Here μ̂_matrix is the matrix we should calculate the column means from.
function standardscale(X, μ̂_matrix; divide_by_sd=true)
    X_copy = copy(X)
    # Standard scale
    for i in 1:size(X, 2)
        μ̂ = mean(μ̂_matrix[:, i])
        X_copy[:, i] .-= μ̂
        if divide_by_sd
            X_copy[:, i] ./= std(μ̂_matrix[:, i])
        end
    end

    return X_copy
end

# Generate a matrix with features as polynomial terms with interactions
function generatedesignmatrix(x1, x2, order)
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

    return X
end

function shufflematrices(X, y)
    shuffleindices = shuffle(1:size(X, 1))
    return X[shuffleindices, :], y[shuffleindices]
end

# Generate a design matrix consisting of random x-s (two explanatory variables)
# with a polynomial of a given order (we will use order 5 for our analysis
# mainly). Additionally includes options for adding intercept column, noise,
# number of observations and random seed.
function generatedata(order::Int64; split=true, include_intercept=false, add_noise=false, noise_σ=0.1, n=200, custom_seed=1000)
    # Setting the seed for train test splitting and random x1/x2
    seed!(custom_seed)

    # Generating x-s
    x1 = rand(n)
    x2 = rand(n)

    # Creating the design matrix
    X = generatedesignmatrix(x1, x2, order)

    # Creating the response
    y = Functions.frankefunction(x1, x2)


    # Shuffle before train-test splitting
    Xs, ys = shufflematrices(X, y)

    # Train-test splitting
    if !split
        return standardscale(Xs, Xs), ys
    end
    indextosplitat = convert(Int, floor(size(Xs, 1) * 0.8))
    X_train, X_test = Xs[1:indextosplitat, :], Xs[(indextosplitat+1):size(Xs, 1), :]
    y_train, y_test = ys[1:indextosplitat, :], ys[(indextosplitat+1):size(ys, 1), :]

    # Scaling X_train/X_test
    # Here we create a copy of the X_train matrix as we want to use the
    # column-means of this to scale both the X_train and X_test columns.
    X_train_original = copy(X_train)
    X_train = standardscale(X_train, X_train_original)
    # We use the column means from the original X_train to subtract from the
    # columns in X_test.
    X_test = standardscale(X_test, X_train_original)

    if add_noise
        # Add response with noise
        y_train += rand(Normal(0, noise_σ), length(y_train))
        y_test += rand(Normal(0, noise_σ), length(y_test))
    end

    if include_intercept
        X_train = [ones(size(X_train, 1)) X_train]
        X_test = [ones(size(X_test, 1)) X_test]
    end

    return X_train, X_test, y_train, y_test
end

end