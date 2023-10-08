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
MSE train: 0.001046, MSE test: 0.007677
R^2 train: 0.985926, R^2 test: 0.907600
With noise
MSE train: 0.008400, MSE test: 0.058654
R^2 train: 0.905406, R^2 test: 0.395051
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
 0.0263635   0.0238954
 0.0165702   0.0168492
 0.0127555   0.0124924
 0.010646    0.0115077
 0.00963469  0.0111353
 0.00921852  0.0115562
 0.00891175  0.0118917
 0.00865817  0.012293
 0.00856731  0.0122392
 0.00853027  0.0123789
 0.0084572   0.0127443
 0.00840279  0.012685
 0.00833427  0.0128101
```

## ResamplingCV.jl
```{txt}
{{conda install-message caused by ScikitLearn.jl}}
{{Lots of convergence warnings}}
k=5
mses for (ols):
13×2 Matrix{Float64}:
 0.179708  0.182166
 0.17028   0.174615
 0.165944  0.175626
 0.16316   0.179613
 0.161615  0.189599
 0.159982  0.239364
 0.158221  0.421732
 0.156493  0.843713
 0.155652  1.18454
 0.154656  1.67863
 0.153797  2.04017
 0.151514  3.23487
 0.14994   4.38114
mses for (ridge):
13×2 Matrix{Float64}:
 0.0168138   0.0171923
 0.00778605  0.00803129
 0.00424517  0.00583649
 0.00267181  0.00543482
 0.0022129   0.00551207
 0.00251701  0.00994649
 0.0027127   0.0146753
 0.00285857  0.0233436
 0.00304985  0.0363649
 0.00326669  0.0568461
 0.00353723  0.0875759
 0.00388968  0.129037
 0.00431031  0.183
mses for (lasso):
13×2 Matrix{Float64}:
 0.0166052   0.0168759
 0.00735677  0.00764859
 0.00582763  0.00608555
 0.00415136  0.00441732
 0.00340088  0.00362233
 0.00312761  0.00331613
 0.00301827  0.00319773
 0.00286608  0.00303419
 0.00268713  0.00285103
 0.00252608  0.00269114
 0.00242658  0.00259712
 0.00236366  0.00253585
 0.00232815  0.00249959
{{Lots of convergence warnings}}
k=10
{{Lots of convergence warnings}}
mses for (ols):
13×2 Matrix{Float64}:
 0.179853  0.181813
 0.170503  0.174631
 0.166432  0.174899
 0.163954  0.178521
 0.162882  0.188791
 0.162053  0.2432
 0.161184  0.419358
 0.160301  0.896935
 0.159859  1.28567
 0.159319  1.87294
 0.158894  2.28614
 0.157933  3.49679
 0.157137  4.79097
mses for (ridge):
13×2 Matrix{Float64}:
 0.0167227   0.0168261
 0.00758179  0.00775823
 0.00384594  0.0049848
 0.00197222  0.00454624
 0.00135025  0.00483685
 0.00136603  0.00890204
 0.00141955  0.0148903
 0.00143702  0.025693
 0.00150918  0.0413329
 0.00161424  0.0649618
 0.00173552  0.099993
 0.00188091  0.147401
 0.00205132  0.20929
mses for (lasso):
13×2 Matrix{Float64}:
 0.0166179   0.0169063
 0.0073746   0.007608
 0.00583914  0.00606723
 0.00416163  0.00438097
 0.00340756  0.00358167
 0.00313476  0.00327898
 0.00302581  0.00316314
 0.0028718   0.00300319
 0.00269211  0.00282286
 0.00253225  0.00266788
 0.0024336   0.00257725
 0.00237124  0.00251951
 0.00233479  0.00248431
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
OLS
Train: 1.606986551239298e-5, test: 1.6075296488878954e-5
Test-data mse for ols: 6.588165084329605e-5
Ridge
Best λ (mse, train): 2.154435e-09, mse: 0.000016
Best λ (mse, test): 2.154435e-09, mse: 0.000016
10×2 Matrix{Float64}:
 1.60727e-5  1.60787e-5
 1.60727e-5  1.60787e-5
 1.60727e-5  1.60787e-5
 1.60727e-5  1.60787e-5
 1.60852e-5  1.60913e-5
 1.62398e-5  1.62461e-5
 1.65439e-5  1.65502e-5
 1.75348e-5  1.75391e-5
 1.96554e-5  1.96572e-5
 2.33015e-5  2.33038e-5
Test-data mse for optimal λ: 1.5878386991627745e-5
Lasso
Best λ (mse, train): 1.000000e-10, mse: 0.000024
Best λ (mse, test): 1.000000e-10, mse: 0.000024
10×2 Matrix{Float64}:
 2.39159e-5  2.39184e-5
 2.3916e-5   2.39185e-5
 2.39177e-5  2.39203e-5
 2.39591e-5  2.39616e-5
 2.51414e-5  2.51435e-5
 2.83425e-5  2.83445e-5
 4.95753e-5  4.95756e-5
 4.95753e-5  4.95756e-5
 4.95753e-5  4.95756e-5
 4.95753e-5  4.95756e-5
{{convergence-warning}}
Test-data mse for optimal λ: 2.365336207037299e-5
```