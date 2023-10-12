using StatsBase: sample
using LinearAlgebra
using Statistics
using Plots

include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score

function bootstrapbiasvariance(orders, n, B=50)
    train_mses = zeros(length(orders))
    test_mses = zeros(length(orders))
    for (i, order) in enumerate(orders)
        X_train, X_test, y_train, y_test = generatedata(order, include_intercept=true, add_noise=true, n=n, custom_seed=1234)

        mse_train = zeros(B)
        mse_test = zeros(B)

        for j in 1:B
            bootstrapsampleindices = sample(collect(1:size(X_train, 1)), size(X_train, 1))
            X_train_bootstrap = X_train[bootstrapsampleindices, :]
            y_train_bootstrap = y_train[bootstrapsampleindices]

            β̂ = pinv(X_train_bootstrap' * X_train_bootstrap) * X_train_bootstrap' * y_train_bootstrap
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

# Here we use 1000 observations (in contrast to the 200 which has been used in
# the other parts of our report) as we use up to 12th order polynomials, which
# means we get 1+12+12+11^2=146 features, and with train/test-splitting we end
# up with almost a many features as observations, and we get a pretty much
# singular matrix (which the pinv seems to have problems computing the inverse
# for). To solve this problem we simply add more observations.
train_mses, test_mses = bootstrapbiasvariance(collect(2:14), 1000)
# First column train mses, second column test mses for each order
display([train_mses test_mses])
plot([collect(2:14), collect(2:14)],
    [train_mses, test_mses],
    label=["train" "test"],
    xlabel="order",
    ylabel="Bootstrap MSE")
savefig(dirname(@__DIR__) * "/figures/bootstrapbiasvariance.png")