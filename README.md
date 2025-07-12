# Deep_Learning_Skills

Deep Learning - Subset of Machine Learning; Based on Multi-Layered Neural Networks; We have both Supervised and Unsupervised Techniques

**Main Taks which we created: To Mimic Human Brain (We want Machines to think Like Human)**

**Frameworks: Tensorflow**

**Contents:**

**1. Types of DL**

**2. Why DL is so popular?**

**3. Perceptron - Intuition**

**4. Advantages and Disadvantages of Perceptron**

**5. ANN - Intuition and Learning**

**6. Backward Propagation and Weight Updation**

1. **Types:**

**1. Artificial Neural Network (ANN):** Solves both Classification and Regression

**2. Convolutional Neural Network (CNN):** Solves better with Image and Video Frames

RCNN, Masked RCNN helps to do Object Detection, along with Image Segmentation

**3. Recurrent Neural Network:** Works well for NLP, Time Series (Time Series like Forecasting Sales)

Better for which involves Sequence of Data

We will learn about:

**(i) Word Embedding**

**(ii) LSTM RNN**

**(iii) GRU RNN**

**(iv) Bidirectional LSTM RNN**

**(v) Encoder Decoder**

**(vi) Transformers**

**(vii) BERT**

2. **Why Deep Learning Getting Popular?**

In 2005, Facebook came into Picture (During 2005-06 also DL was into Picture, but it isn't that Popular). Later around 2011-12, most of the social media platforms comes into picture like Whatsapp, LinkedIn, and hence more data got generated. Companies got huge data, so they don't want to just store data, but they want to use it intelligently. Then due to huge data, "Big Data" cames into picture and thus Hadoop was introduced.

By 2011-12, it became very popular.

In 2011, Cloudera gave Hadoop VM's

(i) Hardware Requirement: NVDIA came up with cost efficient GPU's

(ii) Huge amount of Data --> Deep Learning Models perform very well when compared to traditional Machine Learning Models.

(iii) Used in many industries like Medicine, Ecommerce, Retail, Marketing

(iv) Opensource Frameworks leads to more community users and thus more researches came.

**Tensorflow (Google)**

**Pytorch (Facebook)**

3. **Perceptron Intuition**

Perceptron - Simple Neural Network / Single Layered Neural Network; Basic Form of ANN (Has Input Layer, Hidden Layer, Weights, Activation Function)

**Used for Binary Classification**

**Single Layered because we have only one hidden layer and we don't count the Input Layer**

**(i) Input Layer: No. the Inputs / Features**

**Weight is similar to that, If I keep a hot object in hand, I will take it; These signals are done through Neurons**

**Weights are similar to that, so that the Neurons get activated**

**Bias: Important because, Weights are getting initialized randomly. What if it goes to zero and no process doesn't happen. Weight is treated as noise, meaning, it is a completely new, which our model can handle (It is similar to intercept in Linear Regression)**

**(ii) Hidden Layer: Has Two Steps**

**Step 1: Summation of all weights*Features + Bias (z=∑ wixi​+b)**

**Step 2: Applying Activation Function on Z; Because we want it to come to certain value; Say 0 or 1 here**

Types of Activation Function:

a) Step Function: Threshold Value - 0; Output - 0 if z<=0 or 1 z>0

b) Sigmoid Function: Threshold Value - 0.5; Output - 1 if z>0.5 or 0 if z<=0.5

Perceptron - We are able to create linear classifier line.

z=wixi+b is similar to Linear Regression equation, y=B0 + B1x1 + B2x2 +...+Bnxn

**4. Advantages and Disadvantages of Perceptron**

Perceptron Models are of Two Types: Single Layered Perceptron Model, Multi-Layered Perceptron Model

**Multi Layered Perceptron Models also known as ANN or Multi-Layered Neural Network**

**Problem with Single Layered Perceptron Model:**

In Single Layered Perceptron Model, we have only Feed forward Neural Network and update the weights randomly if error is high (And the process continues). There is no mechanism to update the weights on some technique. (It is best only for Linearly Separable Usecases)

**In Multi-Layered Neural Network, we have:**

**(i) Forward Propagation**

***(ii) Backward Propagation**

**(iii) Loss Function**

**(iv) Activation Function**

**(v) Optimizers**

**Multi-Layered Perceptron Models can be created into Deep Layered Neural Network, which can solve more complex problems**

**5. ANN - Intuition and Learning:**

Also known as Multi-Layered Perceptron / Multi-Layered Neural Network (More than 1 Hidden Layer; We can have any number of Hidden Layers / any number of Neurons in the Hidden Layer)

If 2 Layers, Two-Layered Neural Network

In Hidden Layer, Two Steps Happen: Calculation of 'z', Applying Activation Function for 'z'

**(i) Forward Propagation: We apply Step 1 and 2 in every hidden neuron (Say Random Weight Initialization, calculating z, activation(z) for that layer); Same way if goes through other hidden layers, and as it goes along weights will be added (Say W1-W3 weights in Input Layer, after output O1 at Hidden Layer 1, Weight W4 will be added to it, So it will be O1*W4+b2 (Very Important)**

**Let's say finally by this, we came to the Output Layer; At the Output Layer, we calculate the Loss Function (Loss) with respect to the Actual and the Predicted Value. If it's not matching, error is there and our aim is to reduce the error**

**How to Minimize the Error (Backward Propagation): Updating the weights is the only option; Weight Updatipon goes in Backwards; Let's say first W4 will get updated, then W1,W2,W3. This is known as Backward Propagation**

**Sigmoid Function: 1/(1+e^-z)** 

**Once Backward Propagation is done and all weights are updated, then we send the next record or the inputs**

**Subsequently, the same way in the Hidden and the Output Layers, same process happen. Again loss will be calculated, with respect to that Output.**

Based on Activation Function, Neuron should get activated, It becomes important in Activating/Deactivating (Deactivating - If I keep hot object in right hand, left hand neuron has no role)

**Loss function vs Cost Function: Loss Function is with respect to single point (y-y^)2; Cost Function is with respect to whole data points (Summ i=n (y-y^)2**

**In Backward Propagation, weight updation and Minimising Loss is done through "Optimizers"**

**6. Backward Propagation and Weight Updation:**

**Number of neurons in the Input Layer will be equal to the number of Input Features. Any Number of Neurons can be in Hidden Layer and any number of hidden layers we can have**

**b1,b2 will be Bias and we generally initiate the value of 1; When One Hidden Layer and One Output Layer --> Two Layered Neural Network**

**Output Layer has one Neuron only because it is a simple classification problem (0 or 1). We can have 2 Neurons in the output layer if we have a Multi-Class Classification.**

**Since 3 Neurons in the Input Layer and 2 Neurons in Hidden Layer, we have a Matrix Multiplication of 3x2 Matrix**

**Loss Function: It is how we calculate Loss. We have different Loss functions for different types** 

**Loss Functions for Regression: MSE, MAE, Huber Loss**

**Loss Functions for Classification: Binary Cross Entropy, Categorical Cross Entropy**

Simple Loss Function --> (y-y^)2

**Our Main Aim is to Reduce the Loss Function**

If More Error --> Update the weights by Backward Propagation (Subtracting the Weights and Updating it)

**Weight Updation Formula:**

Wnew = Wold - n (dL/dWold)

The Graph that we plot is Gradient Descent, with respect to Weight and Loss Function.

**To Minimse Loss, we use Optimizer**

**In this case, it is known as Gradient Descent Optimizer**

**If Right side of Slope upwards, Positive Slope; Right side downwards Negative Slope**

**If Downwards - Negative Slope; Wnew > Wold**

**If Upwards - Positive Slope; Wnew < Wold**

**Reducing Weights to come to Global Minima, where loss is minimal**

**n --> Learning Rate (Generally start with 0.001)**

**When optimizer has to Stop:** When we reach the Global Minima, Slope will be 0 --> Wnew = Wold 

So, we no need to update the weights and loss also comes down; As we go down in the chart itself, going to Global Minima, Loss Automatically reduces
