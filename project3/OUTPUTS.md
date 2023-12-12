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
AdaGrad best: 0.9773
Adam best: 0.9769
RMSProp best: 0.98295
SGD best: 0.97985
====Activation functions====
relu: 0.98
sigmoid: 0.96545
swish: 0.98075
tanh: 0.96885
elu: 0.97125


RMSProp relu best hyperparams (100, 100): eta=10^-2.0, lambda=10^-7.0
{(10,): 0.96875, (10, 10, 10): 0.97105, (100, 100): 0.97925, (50, 50): 0.98205, (100, 100, 100, 100): 0.98135, (10, 10, 10, 10): 0.97415, (100, 100, 100, 100, 100): 0.98135, (1000, 100, 10): 0.97855, (1000, 1000): 0.98045, (1000, 1000, 1000): 0.97685, (40, 100, 30): 0.97985}
====Final model accuracy====
0.997298818232977
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
