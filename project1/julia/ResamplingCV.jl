using LinearAlgebra
using Plots
using Statistics
using ScikitLearn: @sk_import, fit!, predict

include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score, kfold, calculateridgeintercept

@sk_import linear_model:Lasso

# We here have chosen a default value of λ to be 10^-5 because this performed
# the best on Lasso, so we may as well reuse this here as we need to specify
# some penalty to use ridge/lasso. One could perhaps fit with many different
# lambdas and choose the best one, but then this metric would be too optimisitc
# (see Hasie et. Al chapter 7).
function crossvalbiasvariance(orders, n, nfolds, λ=10^(-5))
    train_mses_ols = zeros(length(orders))
    test_mses_ols = zeros(length(orders))
    train_mses_ridge = zeros(length(orders))
    test_mses_ridge = zeros(length(orders))
    train_mses_lasso = zeros(length(orders))
    test_mses_lasso = zeros(length(orders))
    kfoldindices = kfold(nfolds, n)
    for (i, order) in enumerate(orders)
        X, y = generatedata(order, split=false, include_intercept=true, add_noise=true, n=n)

        mse_train_ols = zeros(nfolds)
        mse_test_ols = zeros(nfolds)
        mse_train_ridge = zeros(nfolds)
        mse_test_ridge = zeros(nfolds)
        mse_train_lasso = zeros(nfolds)
        mse_test_lasso = zeros(nfolds)

        for j in 1:nfolds
            testindices = kfoldindices[j]
            # trainindices is the complement of testindices (with universe {1,
            # ..., n}), which in julia we can use setdiff to gather this
            trainindices = setdiff(collect(1:n), testindices)

            X_train = X[trainindices, :]
            y_train = y[trainindices]
            X_test = X[testindices, :]
            y_test = y[testindices]

            # Fit/predict ordinary least squares linear regression
            β̂_ols = pinv(X_train' * X_train) * X_train' * y_train
            ŷ_train_ols = X_train * β̂_ols
            ŷ_test_ols = X_test * β̂_ols

            # Fit/predict ridge regression
            β̂_ridge = pinv(X_train' * X_train + λ * I) * X_train' * y_train
            β̂_0 = calculateridgeintercept(X_train, y_train, β̂_ridge)
            ŷ_train_ridge = X_train * β̂_ridge .+ β̂_0
            ŷ_test_ridge = X_test * β̂_ridge .+ β̂_0

            # Fit/predict lasso regression
            model_lasso = fit!(Lasso(alpha=λ, random_state=1234, max_iter=5000), X_train, y_train)
            ŷ_train_lasso = predict(model_lasso, X_train)
            ŷ_test_lasso = predict(model_lasso, X_test)

            # Calculate mses for the different methods
            mse_train_ols[j] = mse(y_train, ŷ_train_ols)
            mse_test_ols[j] = mse(y_test, ŷ_test_ols)
            mse_train_ridge[j] = mse(y_train, ŷ_train_ridge)
            mse_test_ridge[j] = mse(y_test, ŷ_test_ridge)
            mse_train_lasso[j] = mse(y_train, ŷ_train_lasso)
            mse_test_lasso[j] = mse(y_test, ŷ_test_lasso)
        end

        train_mses_ols[i] = mean(mse_train_ols)
        test_mses_ols[i] = mean(mse_test_ols)
        train_mses_ridge[i] = mean(mse_train_ridge)
        test_mses_ridge[i] = mean(mse_test_ridge)
        train_mses_lasso[i] = mean(mse_train_lasso)
        test_mses_lasso[i] = mean(mse_test_lasso)
    end

    return train_mses_ols, test_mses_ols, train_mses_ridge, test_mses_ridge, train_mses_lasso, test_mses_lasso
end

for k in [5, 10]
    # The explanation for using 1000 observations here is the same as the bootstrap
    # one.
    train_mses_ols, test_mses_ols, train_mses_ridge, test_mses_ridge, train_mses_lasso, test_mses_lasso = crossvalbiasvariance(collect(2:14), 1000, k)
    println("k=$(k)")
    for (mses_train, mses_test, method_name) in [
        [train_mses_ols, test_mses_ols, "ols"],
        [train_mses_ridge, test_mses_ridge, "ridge"],
        [train_mses_lasso, test_mses_lasso, "lasso"]]
        println("mses for ($(method_name)):")
        display([mses_train mses_test])
        plot([collect(2:14), collect(2:14)],
            [mses_train, mses_test],
            title="Cross-validation bias-variance ($(method_name)) with $(k) folds",
            label=["train" "test"],
            xlabel="order",
            ylabel="Bootstrap MSE")
        savefig(dirname(@__DIR__) * "/figures/crossvalbiasvariance_$(method_name)__$(k)_folds.png")
    end
end