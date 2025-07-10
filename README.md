# Deep_Learning_Skills

Deep Learning - Subset of Machine Learning; Based on Multi-Layered Neural Networks; We have both Supervised and Unsupervised Techniques

**Main Taks which we created: To Mimic Human Brain (We want Machines to think Like Human)**

**Frameworks: Tensorflow**

**Contents:**

**1. Types of DL**

**2. Why DL is so popular?**

**3. Perceptron - Intuition**

**4. Advantages and Disadvantages of Perceptron**

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
