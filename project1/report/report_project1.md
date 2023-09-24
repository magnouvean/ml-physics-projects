## Introduction

## Rigde expressions
We now look at some properties/values of the ridge regression. We first
calculate the expectation of some $y_i$. We have:
$$\begin{align*}
\mathbb{E}(y_i) &= \mathbb{E}\left[ (\mathbf{X}\mathbf{\beta})_i + \epsilon_i \right] \\
&= \mathbb{E}\left[ (\mathbf{X}\mathbf{\beta})_i \right] + \mathbb{E} \left[ \epsilon_i \right] \\
&= (\mathbf{X}\mathbf{\beta})_i + 0 \\
&= \mathbf{X}_{i, *}\mathbf{\beta}
\end{align*}$$
For the variance get have:
$$\begin{align*}
Var(y_i) &= Var(\mathbf{X}_{i, *} \mathbf{\beta} + \epsilon_i) \\
&= Var(\epsilon_i) \\
&= \sigma^2
\end{align*}$$
(The first to the second line here follows from the fact that $\mathbf{X} \mathbf{\beta}$ is not stochastic).
This is enough to conclude that $y_i \sim N(\mathbf{X}_{i, *} \mathbf{\beta}, \sigma^2)$.
To analyze our $\mathbf{\hat{\beta}}$ we can look at the expectance/variance of
this. In order to elegantly calculate this we use two somewhat basic results (at
the moment I do not prove this). For some non-stochastic matrix $\mathbf{A}$ and some
stochastic $\mathbf{X}$ (with compatible dimesnions) we have the following:
$$\mathbb{E}(\mathbf{A} \mathbf{X}) = \mathbf{A} \mathbb{E}(\mathbf{X})$$
and
$$Var(\mathbf{A} \mathbf{X}) = \mathbf{A} Var(\mathbf{X}) \mathbf{A}^T$$
Using this we get:
$$\begin{align*}
\mathbb{E}(\mathbf{\hat{\beta}}) &= \mathbb{E}((\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}) \\
&= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbb{E}(\mathbf{Y}) \\
&= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{X} \mathbf{\beta} \\
&= \mathbf{\beta}
\end{align*}$$
and
$$\begin{align*}
Var(\mathbf{\hat{\beta}}) &= Var((\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}) \\
&= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T Var(\mathbf{Y}) \left( (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \right)^T \\
&= (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \sigma^2 I \mathbf{X} \left( (\mathbf{X}^T \mathbf{X})^{-1} \right)^T \\
&= \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{X} \left((\mathbf{X}^T \mathbf{X})^T \right)^{-1} \\
&= \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \\
&= \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}
\end{align*}$$

## Bias variance expressions
We assume $\mathbf{y} = f(\mathbf{x}) + \epsilon$, and $\mathbf{y} =
X\hat{\beta} = \hat{f}(\mathbf{x})$. We want to derive the bias-variance
equation. We have:
$$\begin{align*}
E((\mathbf{y} - \mathbf{\hat{y}})^2) &= E((f(\mathbf{x}) + \epsilon - \hat{f}(\mathbf{x}))^2) \\
&= E(\epsilon^2 + 2\cdot \epsilon \cdot (f(\mathbf{x}) - \hat{f}(\mathbf{x})) + (f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2) \\
&= E(\epsilon^2) + 2\cdot E(\epsilon) \cdot E(f(\mathbf{x}) - \hat{f}(\mathbf{x})) + E((f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2) \\
&= E((\epsilon - E(\epsilon))^2) + E((f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2) \\
&= \sigma^2 + E((f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2) \\
\end{align*}$$
Where the last equality follows from the fact that $\epsilon$ (the residuals) is
assumed to be independent of the value of $\hat{f}(\mathbf{x})$. We now
concentrate on calculating the second term. We have:
$$\begin{align*}
E\left[(f(\mathbf{x}) - \hat{f}(\mathbf{x}))^2\right] &= E\left[((f(\mathbf{x}) - E(\hat{f}(\mathbf{x}))) + (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x})))^2\right] \\
&= E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2 + 2\cdot (f(\mathbf{x}) - E(\hat{f}(\mathbf{x}))) \cdot (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x})) + (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))^2\right] \\
&= E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2\right] + 2\cdot E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x}))) \cdot (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))\right] + (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))^2 \\
&= E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2\right] + 2\cdot E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))\right] \cdot E\left[(E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))\right] + (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))^2 \\
&= E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2\right] + 2\cdot E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))\right] \cdot (E(\hat{f}(\mathbf{x})) - E(\hat{f}(\mathbf{x}))) + (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))^2 \\
&= E\left[(f(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2\right] + (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))^2 \\
\end{align*}$$

If we denote $Bias(\hat{y}) = (E(\hat{f}(\mathbf{x})) - \hat{f}(\mathbf{x}))^2 = (\hat{f}(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2$,
and $var(\hat{y}) = E((\hat{f}(\mathbf{x}) - E(\hat{f}(\mathbf{x})))^2)$ we get:
$$E((\mathbf{y} - \mathbf{\hat{y}})^2) = \sigma^2 + Bias(\hat{\mathbf{y}}) + var(\hat{\mathbf{y}})$$
Which shows us the bias-variance tradeoff.

Here $\sigma^2$ is the irreducible part of the MSE, which comes directly from
the data. No matter how good of a model we create this will always be a source
of mean squared error. The bias/variance however is dependent on our model. The
bias can be seen as the long-run sum of difference between the true model $f$
and our model $\hat{f}$ (we can also see it as how much we routinely miss). The
variance of the model is how much the predictions differ from each other.