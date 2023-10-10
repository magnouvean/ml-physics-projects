using LinearAlgebra
using Plots

# Local modules
include("./Data.jl")
include("./Functions.jl")

using .Data: generatedata
using .Functions: mse, r2score

function linearregression(X_train, X_test, y_train, y_test)
    β̂ = pinv(X_train' * X_train) * X_train' * y_train
    ŷ_train = X_train * β̂
    ŷ_test = X_test * β̂
    return mse(y_train, ŷ_train), mse(y_test, ŷ_test), r2score(y_train, ŷ_train), r2score(y_test, ŷ_test), β̂
end

function calculate_scores(orders, add_noise)
    mse_scores_train = zeros(length(orders))
    r2_scores_train = zeros(length(orders))
    mse_scores_test = zeros(length(orders))
    r2_scores_test = zeros(length(orders))
    β_avg_size = zeros(length(orders))

    for (i, k) in enumerate(orders)
        X_train, X_test, y_train, y_test = generatedata(k, include_intercept=true, add_noise=add_noise)
        mse_train, mse_test, r2_train, r2_test, β̂ = linearregression(X_train, X_test, y_train, y_test)

        mse_scores_train[i] = mse_train
        mse_scores_test[i] = mse_test
        r2_scores_train[i] = r2_train
        r2_scores_test[i] = r2_test
        β_avg_size[i] = sum(abs.(β̂)) / length(β̂)
    end

    return mse_scores_train, mse_scores_test, r2_scores_train, r2_scores_test, β_avg_size
end

function plot_scores(orders, mse_train, mse_test, r2_train, r2_test, title, filename)
    figuresdirectory = dirname(@__DIR__) * "/figures/"
    plot([orders, orders],
        [[mse_train r2_train], [mse_test r2_test]],
        label=[["train" "train"] ["test" "test"]],
        xlabel="polynomial order",
        ylabel=["MSE" "R^2"],
        plot_title=title,
        plot_titlefontsize=14,
        layout=2)
    savefig(figuresdirectory * filename * ".png")
end

function show_info(mse_train, mse_test, r2_train, r2_test)
    println("Best performance MSE (train data): $(minimum(mse_train)), order: $(argmin(mse_train))")
    println("Best performance R^2 (train data): $(maximum(r2_train)), order: $(argmax(r2_train))")
    println("Best performance MSE (test data): $(minimum(mse_test)), order: $(argmin(mse_test))")
    println("Best performance R^2 (test data): $(maximum(r2_test)), order: $(argmax(r2_test))")
end

orders = collect(1:5)

println("Without noise")
mse_train, mse_test, r2_train, r2_test, β_sizes_no_noise = calculate_scores(orders, false)
plot_scores(orders, mse_train, mse_test, r2_train, r2_test, "No noise", "linearregression_no_noise")
show_info(mse_train, mse_test, r2_train, r2_test)

println("With noise")
mse_train, mse_test, r2_train, r2_test, β_sizes_with_noise = calculate_scores(orders, true)
plot_scores(orders, mse_train, mse_test, r2_train, r2_test, "With noise", "linearregression_with_noise")
show_info(mse_train, mse_test, r2_train, r2_test)

plot([orders orders],
    [β_sizes_no_noise β_sizes_with_noise],
    label=["without noise" "with noise"],
    xlabel="polynomial order",
    ylabel="avg(abs(β))")
savefig(dirname(@__DIR__) * "/figures/linearregression_beta_size.png")