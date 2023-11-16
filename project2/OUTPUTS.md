## optimization_basic.py
```
Analytical solution: [ 4.00747691e+00  4.64480339e-03 -5.99640873e+00 -7.38378075e-03
  9.95671903e-01]


====Increasing amount of epochs====
COST GD 100 epochs: 4.514991674291281, 400 epochs: 1.5556731277506866
COST SGD 100 epochs: 0.1315013534758629, 400 epochs: 0.03963329873723359


====Varying the SGD size====
MINIBATCH SIZE 8, best cost: 0.042071736388053525
MINIBATCH SIZE 12, best cost: 0.06124977345736559
MINIBATCH SIZE 16, best cost: 0.1315013534758629
MINIBATCH SIZE 32, best cost: 0.7856024224204502


====GD====
Best cost: 4.514991334909431
Best beta: [ 0.77410609 -0.65341437  0.03903951  0.22541462 -0.57386494]


====SGD====
Best cost: 4.109165226574193
Best beta: [ 0.89549959 -0.6526804  -0.25873467  0.22411308 -0.48222584]


====GD with momentum====
Best cost: 0.038397663608929745
Best beta: [ 4.00091553  0.00952591 -5.98956871 -0.01084104  0.99364394]


====SGD with momentum====
Best cost: 0.04875922477494541
Best beta: [ 3.85277295e+00 -4.02256846e-03 -5.71658134e+00 -3.11186857e-04
  9.28595126e-01]


====Adagrad GD====
Best cost: 1.465548118815578
Best beta: [ 2.28403950e+00 -1.13289268e-03 -2.52913022e+00 -5.28463721e-03
  7.39661604e-02]


====Adagrad SGD====
Best cost: 0.0418011880091029
Best beta: [ 3.92432332  0.01080013 -5.85365391 -0.01465819  0.96411616]


====Adagrad GD with momentum====
Best cost: 0.039989275264951024
Best beta: [ 3.98396730e+00 -1.46331079e-03 -5.93432570e+00 -8.86415504e-03
  9.84423041e-01]


====Adagrad SGD with momentum====
Best cost: 0.03851119341067696
Best beta: [ 3.98989272  0.0192844  -5.97828484 -0.01059393  0.9921093 ]
```

## neuralnet_frankefunction.py
```
MSE train best: 0.006788329272142212, at eta=0.0005623413251903491, lambda=0.0001
MSE test best: 0.007157571401982606, at eta=0.0005623413251903491, lambda=0.001
R^2 train best: 0.9243777349795611, at eta=0.0005623413251903491, lambda=0.0001
R^2 test best: 0.9266264019092988, at eta=0.0005623413251903491, lambda=0.001

Sklearn
mse, train (sklearn): 0.09842483663006255
mse, test (sklearn): 0.10908506753060368

Tensorflow

1/5 [=====>........................] - ETA: 0s
5/5 [==============================] - 0s 1ms/step

1/2 [==============>...............] - ETA: 0s
2/2 [==============================] - 0s 1ms/step
mse, train (tensorflow): 0.07256080309601869
mse, test (tensorflow): 0.07957215430922636


====Costs for different activation functions====
sigmoid best MSE: 0.008605771503852864, eta=0.0005623413251903491, lambda=0.01
relu best MSE: 0.017774542401904007, eta=2.3713737056616554e-05, lambda=1.0
lrelu best MSE: 0.016376737331792436, eta=2.3713737056616554e-05, lambda=1.0
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
<function sigmoid at 0x7f7d17df04a0> max acc: 1.0, lr=0.0001
<function relu at 0x7f7d17df05e0> max acc: 0.9773869346733668, lr=0.0001
<function lrelu at 0x7f7d17df0720> max acc: 0.9723618090452262, lr=0.0001

End of training loss:
<function sigmoid at 0x7f7d17df04a0>: 0.0012112684297163737, lr=0.01
<function relu at 0x7f7d17df05e0>: 207.23265815844144, lr=0.01
<function lrelu at 0x7f7d17df0720>: 253.28463078624117, lr=0.001
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
COST: 0.09530791441591768
COST: 0.03270464455050852
COST: 0.017349715004354775
COST: 0.0122125970688519
Convergence after 804 epochs
Final model accuracy: 1.0

1/3 [=========>....................] - ETA: 0s
3/3 [==============================] - 0s 5ms/step
Final model accuracy (tensorflow): 0.9855072463768116
```

## logisticregression_breastcancer.py
```
Best accuracy (train): 0.9949748743718593, lambda=1e-10
Best accuracy (validation): 0.9509803921568627, lambda=1.0


Final model accuracy: 0.9710144927536232
Final model accuracy sklearn: 0.9710144927536232


====Sklearn vs own model betas====
Own model:
[-0.27508581 -0.42983966 -0.28427205 -0.39308309 -0.1989217   0.45048731
 -0.78791301 -0.80778989 -0.02328845  0.40801365 -1.24382468  0.15647918
 -0.63577011 -1.10139002 -0.23784676  0.71176435  0.11720188 -0.44150709
  0.32063682  0.58570076 -0.97774223 -1.15315514 -0.77342976 -1.04002059
 -0.61792775 -0.0162786  -0.7714178  -0.99387819 -0.76331796 -0.74551662]
Sklearn model:
[[-0.2751106  -0.42983294 -0.28429883 -0.3931299  -0.19891633  0.45046368
  -0.78787807 -0.80776894 -0.02327573  0.40803569 -1.24385773  0.15647114
  -0.63576983 -1.10138215 -0.23784667  0.71177714  0.11717816 -0.44151818
   0.32062655  0.58571254 -0.97772824 -1.15315806 -0.77336696 -1.04003541
  -0.61794445 -0.01628587 -0.77140349 -0.99384859 -0.76332908 -0.74554647]]

```
