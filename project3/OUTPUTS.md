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
AdaGrad best: 0.981
Adam best: 0.9802
RMSProp best: 0.9791
SGD best: 0.97785
====Activation functions====
relu: 0.9789
sigmoid: 0.96555
swish: 0.9772
tanh: 0.96845
elu: 0.97305


RMSProp relu best hyperparams (100, 100): eta=10^-2.5, lambda=10^-6.0
{(10,): 0.9605, (10, 10, 10): 0.9649, (100, 100): 0.9781, (50, 50): 0.97485, (100, 100, 100, 100): 0.98195, (10, 10, 10, 10): 0.9676, (100, 100, 100, 100, 100): 0.98115, (1000, 100, 10): 0.98135, (1000, 1000): 0.98145, (1000, 1000, 1000): 0.98195, (40, 100, 30): 0.97885}
====Final model accuracy====
0.9981007048495336
```

## xgb_creditcard.py
```
====Best number of estimators====
240, accuracy: 0.9912

====Best max leaves====
7, accuracy: 0.9923

====Final accuracy====
0.999226213086847
```
