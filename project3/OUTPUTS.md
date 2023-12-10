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
AdaGrad best: 0.97745
Adam best: 0.9779
RMSProp best: 0.98085
SGD best: 0.97705
====Activation functions====
relu: 0.9812
sigmoid: 0.9651
swish: 0.97985
tanh: 0.9692
elu: 0.9728


RMSProp relu best hyperparams (100, 100): eta=10^-2.5, lambda=10^-7.0
{(10,): 0.96015, (10, 10, 10): 0.9663, (100, 100): 0.98015, (50, 50): 0.97875, (100, 100, 100, 100): 0.98275, (10, 10, 10, 10): 0.9685, (100, 100, 100, 100, 100): 0.9851, (1000, 100, 10): 0.9787, (1000, 1000): 0.98405, (1000, 1000, 1000): 0.98135, (40, 100, 30): 0.9792}
====Final model accuracy====
0.9991840180078785
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
