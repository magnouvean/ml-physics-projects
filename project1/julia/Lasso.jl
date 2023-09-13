using ScikitLearn
using MLJLinearModels
using Plots

include("./Data.jl")
include("./Functions.jl")
include("./Ridge.jl")
using .Data
using .Functions: mse, r2score

@sk_import linear_model:Lasso

function calculate_mse_r2_lasso(λs, X_train, X_test, y_train, y_test)
    mse_train = zeros(length(λs))
    mse_test = zeros(length(λs))
    r2_train = zeros(length(λs))
    r2_test = zeros(length(λs))

    for (i, λ) in enumerate(λs)
        model = fit!(Lasso(alpha=λ), X_train, y_train)
        ŷ_train = predict(model, X_train)
        ŷ_test = predict(model, X_test)
        mse_train[i] = mse(y_train, ŷ_train)
        mse_test[i] = mse(y_test, ŷ_test)
        r2_train[i] = r2score(y_train, ŷ_train)
        r2_test[i] = r2score(y_test, ŷ_test)
    end

    return mse_train, mse_test, r2_train, r2_test
end

mse_train, mse_test, r2_train, r2_test = calculate_mse_r2_lasso(λs, X_train, X_test, y_train, y_test)

showinfo(λs, mse_train, mse_test, r2_train, r2_test)

plot([λs, λs], [mse_train, mse_test], label=["train", "test"])
plot([λs, λs], [r2_train, r2_test], label=["train", "test"])