## VBS683 Project

### Test 1 - Activation Functions
* Common Parameters
  * Max Epochs: 300
  * Learning Rate: 0.001
  * Batch Size: 20
  * Accuracy Threshold: 0.25 degree
  * Accuracy Stop Threshold: 1.0
  * Dataset: AL5D_100k
  * Loss Function: Mean Absolute Error


#### Case1 - Sigmoid Function
* Train Time: 580 seconds
* Epochs Number: 252
* Train Loss: 0.2122
* Test Loss: 0.2097
* Mean Error: 0.2097
* Accuracy: 1.0
* Test Errors: 
* 
Mean Error: 1.4018116053193808
X Error: 1.2792784254997969
Y Error: 1.3350880471989512
Z Error: 1.591068459674716
Joint Error
Joint 1 Error: 0.6846312685322212
Joint 2 Error: 0.8685092991382204
Joint 3 Error: 0.7744562068849696
Joint 4 Error: 43.819467940999495

#### Case2 - Tanh Function
* Train Time: 685 seconds
* Epochs Number: 300
* Train Loss: 0.2123
* Test Loss: 0.2124
* Mean Error: 0.2124
* Accuracy: 0.5515
* Test Errors: 

Mean Error: 2.2377732675522566
X Error: 2.6383595541119576
Y Error: 1.534604001790285
Z Error: 2.540356246754527
Joint Error
Joint 1 Error: 0.7154998815992707
Joint 2 Error: 1.1135343848453472
Joint 3 Error: 1.5507567782892542
Joint 4 Error: 43.40239268461642

#### Case3 - ReLU Function
* Train Time: 688 seconds
* Epochs Number: 300
* Train Loss: 0.2246
* Test Loss: 0.2234
* Mean Error: 0.2234
* Accuracy: 0.3958
* Test Errors: 

Mean Error: 3.109402721747756
X Error: 2.9858071357011795
Y Error: 2.837955253198743
Z Error: 3.5044453106820583
Joint Error
Joint 1 Error: 1.9558159176532772
Joint 2 Error: 1.6364386199410081
Joint 3 Error: 1.4448528809392083
Joint 4 Error: 45.63897910947212


### Test 2 - Loss Functions
* Common Parameters
* Max Epochs: 300
* Learning Rate: 0.001
* Batch Size: 20
* Accuracy Threshold: 0.25 degree
* Accuracy Stop Threshold: 1.0
* Dataset: AL5D_100k
* Activation Function: Sigmoid

#### Case1 - Mean Squared Error
* Train Time: 640 seconds
* Epochs Number: 300
* Train Loss: 0.2088
* Test Loss: 0.2064
* Mean Error: 0.2169
* Accuracy: 0.3719

#### Case2 - Mean Absolute Error
* Train Time: 650 seconds
* Epochs Number: 292
* Train Loss: 0.2105
* Test Loss: 0.2076
* Mean Error: 0.2076
* Accuracy: 1.0
