## lr_creditcard.py
```
Logistic regression (CV) with L1 regularization
Accuracy: 0.9652569675994316
Logistic regression (CV) with L2 regularization
Accuracy: 0.9654539315409615
```

## nn_creditcard.py
```
====Model performances====
AdaGrad best: 0.9831
Adam best: 0.98275
RMSProp best: 0.9826
SGD best: 0.98305
====Activation functions====
relu: 0.98245
sigmoid: 0.96595
swish: 0.9814
tanh: 0.9714
elu: 0.9733


RMSProp relu best hyperparams: eta=10^-2.5, lambda=10^-5.0
{(10,): 0.95695, (10, 10, 10): 0.9631, (100, 100): 0.97695, (50, 50): 0.973, (100, 100, 100, 100): 0.97625, (10, 10, 10, 10): 0.96785, (100, 100, 100, 100, 100): 0.97575, (1000, 100, 10): 0.9829, (1000, 1000): 0.9847, (1000, 1000, 1000): 0.97905, (40, 100, 30): 0.97895}
====Final model accuracy====
0.9982554867754643
```
