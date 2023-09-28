using Images
using Plots
using ScikitLearn: @sk_import, fit!, predict
using LinearAlgebra
using Statistics: mean

include("./Data.jl")
include("./Functions.jl")

using .Data: generatedesignmatrix, standardscale, shufflematrices
using .Functions: calculateridgeintercept, kfold, mse

@sk_import linear_model:Lasso

# We will not use the whole data since this will take too long to train using
# any sizable fold/λs. We use the first 400x500 grid of values here.
landscapedata_norway = load(dirname(@__DIR__) * "/data/SRTM_data_Norway_2.tif")[1:400, 1:500]

function datafromlandscape(landscapedata)
    x_indices = collect(1:size(landscapedata, 1))
    y_indices = collect(1:size(landscapedata, 2))

    x = zeros(length(x_indices) * length(y_indices))
    y = zeros(length(x))
    z = zeros(length(x))

    for i in 1:length(x_indices)
        for j in 1:length(y_indices)
            indx = (i - 1) * length(y_indices) + j
            x[indx] = x_indices[i]
            y[indx] = y_indices[j]
            z[indx] = landscapedata_norway[i, j]
        end
    end

    return x, y, z
end

x1, x2, y = datafromlandscape(landscapedata_norway)
# We fix the order of the data to be 5 in our case
X = generatedesignmatrix(x1, x2, 5)
# Scale and shuffle matrix, which here is especially important because of the
# ordered nature and high value of the features.
X = standardscale(X, X)
X, y = shufflematrices(X, y)

function crossvalolsridgelasso(X, y, λs, nfolds)
    train_mses_ols = zeros(nfolds)
    test_mses_ols = zeros(nfolds)
    train_mses_ridge = zeros((length(λs), nfolds))
    test_mses_ridge = zeros((length(λs), nfolds))
    train_mses_lasso = zeros((length(λs), nfolds))
    test_mses_lasso = zeros((length(λs), nfolds))
    n = size(X, 1)
    kfoldindices = kfold(nfolds, n)

    for i in 1:nfolds
        println("Fold: $(i)")
        testindices = kfoldindices[i]
        # trainindices is the complement of testindices (with universe {1,
        # ..., n}), which in julia we can use setdiff to gather this
        trainindices = setdiff(collect(1:n), testindices)

        X_train = X[trainindices, :]
        y_train = y[trainindices]
        X_test = X[testindices, :]
        y_test = y[testindices]

        X_train_with_intercept = [ones(size(X_train, 1)) X_train]
        X_test_with_intercept = [ones(size(X_test, 1)) X_test]

        # Fit/predict ordinary least squares linear regression
        β̂_ols = pinv(X_train_with_intercept' * X_train_with_intercept) * X_train_with_intercept' * y_train
        ŷ_train_ols = X_train_with_intercept * β̂_ols
        ŷ_test_ols = X_test_with_intercept * β̂_ols

        # Fill in MSEs for the folds
        train_mses_ols[i] = mse(y_train, ŷ_train_ols)
        test_mses_ols[i] = mse(y_test, ŷ_test_ols)

        for (j, λ) in enumerate(λs)
            # Fit/predict ridge regression
            β̂_ridge = pinv(X_train' * X_train + λ * I) * X_train' * y_train
            β̂_0 = calculateridgeintercept(X_train, y_train, β̂_ridge)
            ŷ_train_ridge = X_train * β̂_ridge .+ β̂_0
            ŷ_test_ridge = X_test * β̂_ridge .+ β̂_0

            # Fit/predict lasso regression
            model_lasso = fit!(Lasso(alpha=λ, random_state=1234), X_train, y_train)
            ŷ_train_lasso = predict(model_lasso, X_train)
            ŷ_test_lasso = predict(model_lasso, X_test)

            # Fill in the mses for ridge/lasso
            train_mses_ridge[j, i] = mse(y_train, ŷ_train_ridge)
            test_mses_ridge[j, i] = mse(y_test, ŷ_test_ridge)
            train_mses_lasso[j, i] = mse(y_train, ŷ_train_lasso)
            test_mses_lasso[j, i] = mse(y_test, ŷ_test_lasso)
        end
    end

    return mean(train_mses_ols), mean(test_mses_ols), mean(train_mses_ridge, dims=2), mean(test_mses_ridge, dims=2), mean(train_mses_lasso, dims=2), mean(test_mses_lasso, dims=2)
end

λs = 10.0 .^ (range(-14, 0, 12))
ols_train, ols_test, ridge_train, ridge_test, lasso_train, lasso_test = crossvalolsridgelasso(X, y, λs, 3)
println("OLS")
display(ols_train)
display(ols_train)
println("Ridge")
display(ridge_train)
display(ridge_test)
println("Lasso")
display(lasso_train)
display(lasso_test)