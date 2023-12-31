module Functions
using Printf: @printf

using Statistics
using Plots

export frankefunction

function frankefunction(x, y)
    term1 = 0.75 * exp.(-0.25 * (9 * x .- 2) .^ 2 - 0.25 * (9 * y .- 2) .^ 2)
    term2 = 0.75 * exp.(-((9 * x .+ 1) .^ 2) / 49.0 - 0.1 * (9 * y .+ 1))
    term3 = 0.5 * exp.(-((9 * x .- 7) .^ 2) / 4.0 - 0.25 * (9 * y .- 3) .^ 2)
    term4 = -0.2 * exp.(-(9 * x .- 4) .^ 2 - (9 * y .- 7) .^ 2)
    return term1 + term2 + term3 + term4
end

mse(y, ŷ) = mean((y - ŷ) .^ 2)
r2score(y, ŷ) = 1 - (mean((y - ŷ) .^ 2)) / (mean((y .- mean(y)) .^ 2))

# Shows info about some metric, given the lambdas used for the model, the metric
# data on train/test and the metric name. Print this to console in a more easily
# readable format. This will be useful multiple places throughout our code.
function showinfo(λs, metrics_train, metrics_test, metric_name; use_min=true)
    best_λ_train, best_λ_test, best_train, best_test = if use_min
        λs[argmin(metrics_train)], λs[argmin(metrics_test)], minimum(metrics_train), minimum(metrics_test)
    else
        λs[argmax(metrics_train)], λs[argmax(metrics_test)], maximum(metrics_train), maximum(metrics_test)
    end

    @printf "Best λ (%s, train): %e, %s: %.6f\n" metric_name best_λ_train metric_name best_train
    @printf "Best λ (%s, test): %e, %s: %.6f\n" metric_name best_λ_test metric_name best_test
end

# Prints the MSE and R^2 metric values given some different mses for different λ
# values typically obtained using ridge/lasso.
function showinfo_mse_r2(λs, mse_train, mse_test, r2_train, r2_test)
    showinfo(λs, mse_train, mse_test, "mse")
    showinfo(λs, r2_train, r2_test, "r2", use_min=false)
end

# Helper function for plotting mse/r2 train and test against each other
function plotinfo(λs, mse_train, mse_test, r2_train, r2_test, filename; title="")
    figuresdirectory = dirname(@__DIR__) * "/figures/"
    plot([log10.(λs), log10.(λs)],
        [[mse_train r2_train], [mse_test r2_test]],
        label=[["train" "train"] ["test" "test"]],
        xlabel="log10(λ)",
        ylabel=["MSE" "R^2"],
        plot_title=title,
        plot_titlefontsize=14,
        layout=2)
    savefig(figuresdirectory * filename * ".png")
end

function calculateridgeintercept(X_train, y_train, β̂)
    n = size(X_train, 1)
    p = size(X_train, 2)
    return mean(y_train) - 1 / n * sum([sum([X_train[i, j] * β̂[j] for j in 1:p]) for i in 1:n])
end

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

end