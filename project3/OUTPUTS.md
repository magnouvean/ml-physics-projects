## datainfo.py
```
Total observations: 568630
Number of non-fraud: Class    284315
dtype: int64
Number of fraud: Class    284315
dtype: int64
```

## lr_creditcard.py
```
Logistic regression (CV) with L1 regularization
Accuracy: 0.9652569675994316
Logistic regression (CV) with L2 regularization
Accuracy: 0.9654539315409615
Logistic regression (CV) with polynomial features
Accuracy: 0.9972073326861661
```

## nn_creditcard.py
```
====Model performances====
AdaGrad best: 0.98345
Adam best: 0.98265
RMSProp best: 0.9824
SGD best: 0.98335
====Activation functions====
relu: 0.9842
sigmoid: 0.9659
swish: 0.98055
tanh: 0.97105
elu: 0.9724


RMSProp relu best hyperparams (100, 100): eta=10^-2.0, lambda=10^-8.0
{(10,): 0.9721, (10, 10, 10): 0.97, (100, 100): 0.9845, (50, 50): 0.9795, (100, 100, 100, 100): 0.98345, (10, 10, 10, 10): 0.972, (100, 100, 100, 100, 100): 0.98215, (1000, 100, 10): 0.97935, (1000, 1000): 0.9792, (1000, 1000, 1000): 0.97815, (40, 100, 30): 0.97825}
====Final model accuracy====
0.998564997186269
```

## xgb_creditcard.py
```
====Best number of estimators====
360, accuracy: 0.9945

====Best max leaves====
6, accuracy: 0.9955

====Final accuracy====
0.9994091164884638
```
