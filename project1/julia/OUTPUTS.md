# Outputs
Running the 4 files meant to be run resulted in the text outputs given below. We
have set random seeds in the programs, so the same outputs should be given on
each run. Keep in mind, where appropriate I have replaced some parts of these
outputs with descriptions of the outputs, where this is marked inside {{}}
notation. This basically only applies to warnings and other junk output to make
the interesting results readable.

## FrankePlot.jl
No output

## LinearRegression.jl
```{txt}
Without noise
Best performance MSE (train data): 0.0010458873706902273, order: 5
Best performance R^2 (train data): 0.9859263783653947, order: 5
Best performance MSE (test data): 0.005326048089194724, order: 3
Best performance R^2 (test data): 0.9358998345216957, order: 3
With noise
Best performance MSE (train data): 0.00840031822164739, order: 5
Best performance R^2 (train data): 0.9054059424495206, order: 5
Best performance MSE (test data): 0.018910967221048437, order: 3
Best performance R^2 (test data): 0.8049535091279812, order: 3
```

## Ridge.jl
```{txt}
Without noise
Best λ (mse, train): 1.000000e-12, mse: 0.001046
Best λ (mse, test): 1.659587e-02, mse: 0.003592
Best λ (r2, train): 1.000000e-12, r2: 0.985926
Best λ (r2, test): 1.659587e-02, r2: 0.956765
With noise
Best λ (mse, train): 1.000000e-12, mse: 0.008400
Best λ (mse, test): 1.513561e+00, mse: 0.017352
Best λ (r2, train): 1.000000e-12, r2: 0.905406
Best λ (r2, test): 1.513561e+00, r2: 0.821028
```

## Lasso.jl
```{txt}
{{conda install-message caused by ScikitLearn.jl}}
Without noise
{{Lots of convergence warnings}}
Best λ (mse, train): 1.000000e-12, mse: 0.003280
Best λ (mse, test): 1.380384e-05, mse: 0.002737
Best λ (r2, train): 1.000000e-12, r2: 0.955866
Best λ (r2, test): 1.380384e-05, r2: 0.967054
With noise
{{Lots of convergence warnings}}
Best λ (mse, train): 1.000000e-12, mse: 0.011319
Best λ (mse, test): 9.120108e-04, mse: 0.016451
Best λ (r2, train): 1.000000e-12, r2: 0.872539
Best λ (r2, test): 9.120108e-04, r2: 0.830329
```

## Resampling.jl
```{txt}
13×2 Matrix{Float64}:
 0.0262166   0.0269016
 0.0171338   0.0193623
 0.0128578   0.0144817
 0.0101363   0.0116086
 0.00931475  0.0110787
 0.00889111  0.0105788
 0.00861805  0.0107826
 0.00843702  0.0118442
 0.00832217  0.0141671
 0.00820758  0.0169197
 0.00811843  0.013647
 0.00805178  0.0147002
 0.0080152   0.0163047
```

## ResamplingCV.jl
```{txt}
{{conda install-message caused by ScikitLearn.jl}}
{{Lots of convergence warnings}}
k=5
mses for (ols):
13×2 Matrix{Float64}:
 0.169408  0.171808
 0.161001  0.166489
 0.156632  0.165474
 0.153802  0.168085
 0.152556  0.172981
 0.151802  0.180353
 0.151004  0.194121
 0.150502  0.207533
 0.149944  0.234086
 0.149705  0.238246
 0.14924   0.257296
 0.14898   0.269201
 0.148669  0.294628
mses for (ridge):
13×2 Matrix{Float64}:
 0.0159985   0.0159849
 0.00816761  0.0083472
 0.00436695  0.0050403
 0.00261535  0.00396629
 0.00204697  0.00373153
 0.00207214  0.00510296
 0.00220929  0.00598986
 0.00226688  0.00716748
 0.00234675  0.0085226
 0.00244576  0.0101351
 0.00255402  0.0123044
 0.00269187  0.0153871
 0.00286891  0.0197435
mses for (lasso):
13×2 Matrix{Float64}:
 0.0157212   0.0159744
 0.00761632  0.00785954
 0.00598262  0.00618569
 0.00435971  0.00454554
 0.00364307  0.00378919
 0.00330211  0.00341471
 0.00311111  0.00321332
 0.00293106  0.0030277
 0.0027596   0.00285264
 0.00261016  0.00270176
 0.00250623  0.00259932
 0.00243201  0.0025251
 0.00239223  0.0024845
{{Lots of convergence warnings}}
k=10
mses for (ols):
13×2 Matrix{Float64}:
 0.169531  0.171836
 0.161318  0.165791
 0.157122  0.164595
 0.154481  0.168189
 0.153496  0.172643
 0.153042  0.177618
 0.152695  0.183204
 0.15246   0.189606
 0.152251  0.200384
 0.152153  0.204423
 0.151985  0.215135
 0.151872  0.22633
 0.151709  0.250375
mses for (ridge):
13×2 Matrix{Float64}:
 0.0158456   0.0160336
 0.00785293  0.00799701
 0.00392156  0.00439377
 0.00198576  0.00318155
 0.00127843  0.0024768
 0.00117918  0.00289392
 0.0011923   0.00344513
 0.00115747  0.00412603
 0.00117907  0.00488102
 0.0012312   0.00576816
 0.00128132  0.00697229
 0.00133642  0.00883094
 0.00140738  0.011866
mses for (lasso):
13×2 Matrix{Float64}:
 0.0157348   0.0159725
 0.00762878  0.00786308
 0.00598927  0.00621373
 0.00436459  0.00455955
 0.00364644  0.00379733
 0.00330482  0.0034235
 0.00311538  0.00322514
 0.00293347  0.00304113
 0.00276191  0.00287039
 0.00261289  0.00272154
 0.00250885  0.00261878
 0.00243467  0.00254501
 0.00239466  0.00250403
```

## Landscape.jl
```{txt}
{{conda install-message caused by ScikitLearn.jl}}
Fold: 1
{{Lots of convergence warnings}}
Fold: 2
{{Lots of convergence warnings}}
Fold: 3
{{Lots of convergence warnings}}
Fold: 4
{{Lots of convergence warnings}}
Fold: 5
{{Lots of convergence warnings}}
Fold: 6
{{Lots of convergence warnings}}
Fold: 7
{{Lots of convergence warnings}}
Fold: 8
{{Lots of convergence warnings}}
OLS
Train: 1.6070461903413518e-5, test: 1.607538927485563e-5
Test-data mse for ols: 1.5878387992286373e-5
Ridge
Best λ (mse, train): 2.154435e-09, mse: 0.000016
Best λ (mse, test): 4.641589e-08, mse: 0.000016
10×2 Matrix{Float64}:
 1.60714e-5  1.60767e-5
 1.60714e-5  1.60767e-5
 1.60714e-5  1.60767e-5
 1.60714e-5  1.60767e-5
 1.60796e-5  1.60848e-5
 1.62249e-5  1.623e-5
 1.64908e-5  1.64957e-5
 1.73707e-5  1.73754e-5
 1.94709e-5  1.9475e-5
 2.29655e-5  2.29697e-5
Test-data mse for optimal λ: 1.5878365538552768e-5
Lasso
Best λ (mse, train): 1.000000e-10, mse: 0.000024
Best λ (mse, test): 1.000000e-10, mse: 0.000024
10×2 Matrix{Float64}:
 2.3916e-5   2.392e-5
 2.39161e-5  2.392e-5
 2.39179e-5  2.39218e-5
 2.39593e-5  2.39632e-5
 2.51416e-5  2.51446e-5
 2.83427e-5  2.83442e-5
 4.95753e-5  4.95759e-5
 4.95753e-5  4.95759e-5
 4.95753e-5  4.95759e-5
 4.95753e-5  4.95759e-5
{{Convergence warning}}
Test-data mse for optimal λ: 2.365336207037299e-5
```