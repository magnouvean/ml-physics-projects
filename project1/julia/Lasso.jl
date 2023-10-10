using ScikitLearn
using Plots

include("./Data.jl")
include("./Functions.jl")
using .Data: generatedata, λs
using .Functions: showinfo_mse_r2, plotinfo, mse, r2score

@sk_import linear_model:Lasso

function calculate_mse_r2_lasso(λs, X_train, X_test, y_train, y_test)
    mse_train = zeros(length(λs))
    mse_test = zeros(length(λs))
    r2_train = zeros(length(λs))
    r2_test = zeros(length(λs))
    β_avg_sizes = zeros(length(λs))

    for (i, λ) in enumerate(λs)
        model = fit!(Lasso(alpha=λ, random_state=1234, max_iter=5000), X_train, y_train)
        ŷ_train = predict(model, X_train)
        ŷ_test = predict(model, X_test)
        mse_train[i] = mse(y_train, ŷ_train)
        mse_test[i] = mse(y_test, ŷ_test)
        r2_train[i] = r2score(y_train, ŷ_train)
        r2_test[i] = r2score(y_test, ŷ_test)
        β_avg_sizes[i] = (abs(model.intercept_[1]) + sum(abs.(model.coef_))) / (1 + length(model.coef_))
    end

    return mse_train, mse_test, r2_train, r2_test, β_avg_sizes
end

println("Without noise")
X_train, X_test, y_train, y_test = generatedata(5)
mse_train, mse_test, r2_train, r2_test, β_sizes_no_noise = calculate_mse_r2_lasso(λs, X_train, X_test, y_train, y_test)
showinfo_mse_r2(λs, mse_train, mse_test, r2_train, r2_test)
plotinfo(λs, mse_train, mse_test, r2_train, r2_test, "lasso_without_noise", title="Without noise")

println("With noise")
X_train, X_test, y_train, y_test = generatedata(5, add_noise=true)
mse_train, mse_test, r2_train, r2_test, β_sizes_with_noise = calculate_mse_r2_lasso(λs, X_train, X_test, y_train, y_test)
showinfo_mse_r2(λs, mse_train, mse_test, r2_train, r2_test)
plotinfo(λs, mse_train, mse_test, r2_train, r2_test, "lasso_with_noise", title="With noise")

plot([log10.(λs) log10.(λs)],
    [β_sizes_no_noise β_sizes_with_noise],
    label=["without noise" "with noise"],
    xlabel="log10(λ)",
    ylabel="abs(avg(β))")
savefig(dirname(@__DIR__) * "/figures/lasso_beta_size.png")