using LinearAlgebra
using Plots
using Statistics

include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score

# A somewhat simple way to split indices up into nfolds as good as possible equal pieces.
function kfold(nfolds, length)
    sizeperfold = floor(length / nfolds)
    rest = length % nfolds
    kfoldindices = Array{Array{Int,1}}(undef, nfolds)
    foldend = 0
    for i in 1:nfolds
        foldstart = foldend + 1
        # If we are one of "rest" last folds we get an extra item, meaning the
        # rest is spread over the "rest" last folds.
        foldend = if nfolds - i < rest
            foldstart + sizeperfold
        else
            foldstart + sizeperfold - 1
        end
        kfoldindices[i] = collect(foldstart:foldend)
    end
    return kfoldindices
end


function crossvalbiasvariance(orders, n, nfolds)
    train_mses = zeros(length(orders))
    test_mses = zeros(length(orders))
    kfoldindices = kfold(nfolds, n)
    for (i, order) in enumerate(orders)
        X, y = generatedata(order, split=false, include_intercept=true, add_noise=true, n=n)

        mse_train = zeros(nfolds)
        mse_test = zeros(nfolds)

        for j in 1:nfolds
            testindices = kfoldindices[j]
            # trainindices is the complement of testindices (with universe {1,
            # ..., n}), which in julia we can use setdiff to gather this
            trainindices = setdiff(collect(1:n), testindices)

            X_train = X[trainindices, :]
            y_train = y[trainindices]
            X_test = X[testindices, :]
            y_test = y[testindices]

            β̂ = pinv(X_train' * X_train) * X_train' * y_train
            ŷ_train = X_train * β̂
            ŷ_test = X_test * β̂
            mse_train[j] = mse(y_train, ŷ_train)
            mse_test[j] = mse(y_test, ŷ_test)
        end

        train_mses[i] = mean(mse_train)
        test_mses[i] = mean(mse_test)
    end

    return train_mses, test_mses
end

for k in [5, 7, 10]
    # The explanation for using 1000 observations here is the same as the bootstrap
    # one.
    train_mses, test_mses = crossvalbiasvariance(collect(2:14), 1000, k)
    println("k=$(k)")
    println("Train mses:")
    display(train_mses)
    println("Test mses:")
    display(test_mses)
    plot([collect(2:14), collect(2:14)],
        [train_mses, test_mses],
        label=["train" "test"],
        xlabel="order",
        ylabel="Bootstrap MSE")
    savefig(dirname(@__DIR__) * "/figures/crossvalbiasvariance_$(k)_folds.png")
end