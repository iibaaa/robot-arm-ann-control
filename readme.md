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

Mean Error: 3.2438759226351976
X Error: 3.3444499131292105
Y Error: 3.480499144643545
Z Error: 2.9066794086247683

#### Case2 - Mean Absolute Error
* Train Time: 650 seconds
* Epochs Number: 292
* Train Loss: 0.2105
* Test Loss: 0.2076
* Mean Error: 0.2076
* Accuracy: 1.0

Mean Error: 1.1453712359070778
X Error: 1.0893478756770492
Y Error: 1.0242402786388993
Z Error: 1.3225253205746412
