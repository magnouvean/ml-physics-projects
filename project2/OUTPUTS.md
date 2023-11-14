## optimization_basic.py
```
Analytical solution: [ 4.00000000e+00 -2.21628271e-14 -6.00000000e+00  8.72218964e-15
  1.00000000e+00]


====GD====
Best cost: 3.6145195272029205
Best beta: [ 1.04375196 -0.52164984 -0.58070093  0.18454152 -0.4031854 ]


====SGD====
Best cost: 3.7730950294258467
Best beta: [ 0.99799675 -0.70304006 -0.49713236  0.24310943 -0.4149234 ]


====GD with momentum====
Best cost: 0.00043493949341673154
Best beta: [ 3.97668743e+00 -2.45992327e-04 -5.95435333e+00  2.44091405e-03
  9.85908747e-01]


====SGD with momentum====
Best cost: 0.003628796572747663
Best beta: [ 3.90665865 -0.01950566 -5.82969189  0.0065724   0.95509905]


====Adagrad GD====
Best cost: 1.4306290222873281
Best beta: [ 2.27460170e+00 -5.69359725e-03 -2.52855968e+00  2.06859689e-03
  7.71726966e-02]


====Adagrad SGD====
Best cost: 4.052390817166087e-05
Best beta: [ 3.99209356e+00  4.38565185e-04 -5.98274183e+00  5.60274580e-04
  9.95487392e-01]


====Adagrad GD with momentum====
Best cost: 0.001555971681741602
Best beta: [ 3.97562075e+00 -5.82982134e-03 -5.93683910e+00 -1.23300753e-03
  9.88263249e-01]


====Adagrad SGD with momentum====
Best cost: 1.950990104610287e-12
Best beta: [ 3.99999833e+00 -1.49751088e-07 -5.99999619e+00 -7.61111529e-09
  9.99999046e-01]
```

## neuralnet_frankefunction.py
```
Convergence after 192 epochs
MSE train best: 0.009988421389059988, at eta=0.0001333521432163324, lambda=0.1
MSE test best: 0.011652887093602429, at eta=0.0001333521432163324, lambda=1.0
R^2 train best: 0.8967189266203182, at eta=0.0001333521432163324, lambda=0.1
R^2 test best: 0.864482723648619, at eta=0.0001333521432163324, lambda=1.0

Sklearn
mse, train (sklearn): 0.42066157941885995
mse, test (sklearn): 0.08938620708289866

Tensorflow

1/4 [======>.......................] - ETA: 1s - loss: 104.4552 - mse: 0.2915
4/4 [==============================] - 1s 3ms/step - loss: 102.1417 - mse: 0.1994

1/4 [======>.......................] - ETA: 0s
4/4 [==============================] - 0s 1ms/step

1/4 [======>.......................] - ETA: 0s
4/4 [==============================] - 0s 2ms/step
mse, train (tensorflow): 0.10751819997659595
mse, test (tensorflow): 0.09749259779361634
Convergence after 192 epochs
sigmoid best MSE: 0.0116529055120261, eta=0.0001333521432163324, lambda=1.0
relu best MSE: 0.14751593225800425, eta=1.333521432163324e-06, lambda=100.0
lrelu best MSE: 0.1501710148619283, eta=1.333521432163324e-06, lambda=100.0
```

## neuralnet_breastcancer.py
```
Convergence after 70 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 21 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
========Hidden layers========
Best accuracies:
(100,): 1.0, eta=0.001
(10, 10): 1.0, eta=0.01
(100, 100): 1.0, eta=0.0001
(50, 10): 1.0, eta=0.001
(10, 10, 10): 1.0, eta=0.001

End of training loss:
(100,): 0.00124639499682563, eta=0.1
(10, 10): 0.01719081210486661, eta=0.01
(100, 100): 0.0012112684297163737, eta=0.01
(50, 10): 0.007495652314046236, eta=0.1
(10, 10, 10): 0.014259115847315415, eta=0.01
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 63 epochs
Convergence after 31 epochs
Convergence after 14 epochs
Convergence after 12 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 19 epochs
Convergence after 28 epochs
Convergence after 13 epochs
Convergence after 12 epochs
Convergence after 11 epochs
Convergence after 11 epochs

========Activation Functions========
Accuracies:
<function sigmoid at 0x7fb606648360> max acc: 1.0, lr=0.0001
<function relu at 0x7fb6066484a0> max acc: 0.9773869346733668, lr=0.0001
<function lrelu at 0x7fb6066485e0> max acc: 0.9723618090452262, lr=0.0001

End of training loss:
<function sigmoid at 0x7fb606648360>: 0.0012112684297163737, lr=0.01
<function relu at 0x7fb6066484a0>: 207.23265815844144, lr=0.01
<function lrelu at 0x7fb6066485e0>: 253.28463078624117, lr=0.001
Convergence after 37 epochs
Convergence after 11 epochs
Convergence after 13 epochs
Convergence after 15 epochs
Convergence after 65 epochs
Convergence after 39 epochs
Convergence after 23 epochs
Convergence after 29 epochs
Convergence after 15 epochs
Convergence after 17 epochs
Convergence after 74 epochs
Convergence after 38 epochs
Convergence after 26 epochs
Convergence after 31 epochs
Convergence after 82 epochs
Convergence after 29 epochs
Convergence after 27 epochs
Convergence after 27 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 102 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 54 epochs
Convergence after 16 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 12 epochs
Convergence after 13 epochs
Convergence after 116 epochs
Convergence after 37 epochs
Convergence after 81 epochs
Convergence after 40 epochs
Convergence after 45 epochs
Convergence after 134 epochs
Convergence after 39 epochs
Convergence after 30 epochs
Convergence after 30 epochs
Convergence after 33 epochs
Convergence after 32 epochs
Convergence after 170 epochs
Convergence after 170 epochs
Convergence after 170 epochs
Convergence after 175 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 11 epochs
Convergence after 139 epochs
Convergence after 97 epochs
Convergence after 63 epochs
Convergence after 12 epochs
Convergence after 12 epochs
Convergence after 149 epochs
Convergence after 100 epochs
Convergence after 65 epochs
Convergence after 12 epochs
Convergence after 12 epochs
Convergence after 12 epochs
SGD max train accuracy: 1.0
SGD max validation accuracy: 0.9509803921568627
Adagrad max train accuracy: 1.0
Adagrad max validation accuracy: 0.9607843137254902
Adam max train accuracy: 1.0
Adam max validation accuracy: 0.9509803921568627
RMSProp max train accuracy: 1.0
RMSProp max validation accuracy: 0.9509803921568627
Best learning_rate for adagrad: 0.1
Best regularization parameter value for adagrad: 1e-07
COST: 673.4364466838488
COST: 1.5665315450805917
COST: 0.6744193581437385
COST: 0.3712362097292693
COST: 0.23526834085242115
COST: 0.15739974864159914
COST: 0.11049545979862346
COST: 0.08649659569121672
COST: 0.07166338804913726
COST: 0.06134325198070681
Final model accuracy: 1.0

1/3 [=========>....................] - ETA: 0s
3/3 [==============================] - 0s 1ms/step
Final model accuracy (tensorflow): 0.9855072463768116
```

## logisticregression_breastcancer.py
```
Best accuracy (train): 0.9170854271356784, lambda=0.1
Best accuracy (test): 0.9019607843137255, lambda=0.1


Final model accuracy: 0.9565217391304348
Final model accuracy sklearn: 0.9565217391304348
```
