using StatsBase: sample
using Statistics
using Plots

include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score

function bootstrapbiasvariance(orders, n=1000, B=10)
    train_mses = zeros(length(orders))
    test_mses = zeros(length(orders))
    for (i, order) in enumerate(orders)
        X_train, X_test, y_train, y_test = generatedata(order, true, true, true, 0.1, n)

        mse_train = zeros(B)
        mse_test = zeros(B)

        for j in 1:B
            bootstrapsampleindices = sample(collect(1:size(X_train, 1)), size(X_train, 1))
            X_train_bootstrap = X_train[bootstrapsampleindices, :]
            y_train_bootstrap = y_train[bootstrapsampleindices]

            β̂ = inv(X_train_bootstrap' * X_train_bootstrap) * X_train_bootstrap' * y_train_bootstrap
            ŷ_train_bootstrap = X_train_bootstrap * β̂
            ŷ_test = X_test * β̂
            mse_train[j] = mse(y_train_bootstrap, ŷ_train_bootstrap)
            mse_test[j] = mse(y_test, ŷ_test)
        end

        train_mses[i] = mean(mse_train)
        test_mses[i] = mean(mse_test)
    end

    return train_mses, test_mses
end

train_mses, test_mses = bootstrapbiasvariance(collect(2:10))
plot([train_mses, test_mses], label=["train" "test"])