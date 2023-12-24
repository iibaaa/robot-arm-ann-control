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
* Mean Error: 1.4018116053193808 
* X Error: 1.2792784254997969 
* Y Error: 1.3350880471989512 
* Z Error: 1.591068459674716

#### Case2 - Tanh Function
* Train Time: 685 seconds
* Epochs Number: 300
* Train Loss: 0.2123
* Test Loss: 0.2124
* Mean Error: 0.2124
* Accuracy: 0.5515
* Mean Error: 2.2377732675522566
* X Error: 2.6383595541119576
* Y Error: 1.534604001790285
* Z Error: 2.540356246754527


#### Case3 - ReLU Function
* Train Time: 688 seconds
* Epochs Number: 300
* Train Loss: 0.2246
* Test Loss: 0.2234
* Mean Error: 0.2234
* Accuracy: 0.3958
* Mean Error: 3.109402721747756
* X Error: 2.9858071357011795
* Y Error: 2.837955253198743
* Z Error: 3.5044453106820583


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
* Mean Error: 3.2438759226351976
* X Error: 3.3444499131292105
* Y Error: 3.480499144643545
* Z Error: 2.9066794086247683

#### Case2 - Mean Absolute Error
* Train Time: 650 seconds
* Epochs Number: 292
* Train Loss: 0.2105
* Test Loss: 0.2076
* Mean Error: 0.2076
* Accuracy: 1.0
* Mean Error: 1.1453712359070778
* X Error: 1.0893478756770492
* Y Error: 1.0242402786388993
* Z Error: 1.3225253205746412


### Test 3 - Dataset
* Common Parameters
* Max Epochs: 500
* Learning Rate: 0.001
* Batch Size: 20
* Accuracy Threshold: 0.25 degree
* Accuracy Stop Threshold: 1.0
* Activation Function: Sigmoid
* Loss Function: Mean Absolute Error

#### Case1 - AL5D_10k
* Train Time: 113 seconds
* Epochs Number: 500
* Train Loss: 0.2223
* Test Loss: 0.2219
* Mean Error: 0.2219
* Accuracy: 0.57 
* Mean Error: 2.840987406671047 
* X Error: 2.658360404893756 
* Y Error: 2.1181374322623014 
* Z Error: 3.7464648485183716

#### Case2 - AL5D_20k
* Train Time: 242 seconds
* Epochs Number: 500
* Train Loss: 0.2143
* Test Loss: 0.2151
* Mean Error: 0.2151
* Accuracy: 0.78 
* Mean Error: 1.857152790762484 
* X Error: 1.836367417126894 
* Y Error: 1.7093458445742726 
* Z Error: 2.0257446449249983

#### Case3 - AL5D_50k
* Train Time: 574 seconds
* Epochs Number: 500
* Train Loss: 0.2129
* Test Loss: 0.2163
* Mean Error: 0.2163
* Accuracy: 0.61 
* Mean Error: 2.299803774803877 
* X Error: 2.2264430299401283 
* Y Error: 1.990519929677248 
* Z Error: 2.6824481319636106

#### Case4 - AL5D_100k
* Train Time: 580 seconds
* Epochs Number: 252
* Train Loss: 0.2122
* Test Loss: 0.2097
* Mean Error: 0.2097
* Accuracy: 1.0
* Mean Error: 1.4018116053193808 
* X Error: 1.2792784254997969 
* Y Error: 1.3350880471989512 
* Z Error: 1.591068459674716




