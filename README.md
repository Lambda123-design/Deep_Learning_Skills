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

**7. Chain Rule of Derivatives**

**8. Vanishing Gradient and Problem with Sigmoid Activation Function**

**9. Sigmoid Activation Function**

**10. Tanh Activation Function**

**11. ReLu Activation Function**

**12. LeakyReLu and Parametric ReLu**

**13. Exponential Linear Unit (ELU)**

**14. Softmax Activation Function**

**15. Which Activation Function when to Apply?**

**16. Loss Function vs Cost Function**

**17. Regression Cost Function**

**18. Which Loss Functions when to use**

**19. Types of Optimizers**

**20. Gradient Descent Optimizer**

**21. Stochastic Gradient Descent (SGD)**

**22. Mini Batch with SGD**

**23. SGD with Momentum**

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

**n --> Helps in Speed of Convergence; Learning Rate (Generally start with 0.001)**

**When optimizer has to Stop:** When we reach the Global Minima, Slope will be 0 --> Wnew = Wold 

So, we no need to update the weights and loss also comes down; As we go down in the chart itself, going to Global Minima, Loss Automatically reduces

**7. Chain Rule of Derivatives**

**dL/dW4new = dL/dO2 * dO2/dW4**

**8. Vanishing Gradient and Problem with Sigmoid Activation Function**

We know that W1new = W1old - n (dL/dW1old)

dL/dw1old = dl/dO31 x dO31/dO21 x dO21/dO11 * dO11/dW1old --> Eq (i)

We know that, O31 going is z

Let's take, dO31/dO21 = d(sig(z)/d(z) x d(z) / dO21

**Derivative of Sigmoid Activation function is, 0<= Sig(z) <=0.25 [By Mathematically Proved]**

So, In eq (i) 

dO31/dO21 x dO21/dO11 * dO11/dW1old (Each will be value between 0-0.25, Multiplied with each other)

Again in the end in substituting in W1new = W1old - n dL/dW1old, the n(dL/dW1old) becomes very very small, where again n(Learning Rate starts from 0.01, smaller value)

**This in turns means that W1new = W1old (Approximately), which means no weight updation happens to reach Global Minima**

**We are stuck in Weight Updation. This is because of Sigmoid Activation Function, where dervative is between range 0-0.25**

**In a very Deep NN, this will be very negligible values, say 0.00001. So we can't use Sigmoid Activation Function there. For smaller Activation function, we can maybe consider this**

**Researchers came with other activation functions such as: Tanh, Relu, PreRelu, Swiss**

**9. Sigmoid Activation Function:**

Better with smaller NN; When NN goes deep, it didn't work; It was able to show even firing of Neuron

**In ML, especially, Using Standard Scalar was able to convert Data Points to Zero-Centered, with Mean=0 and Standard Deviation = 1, which in similar in DL, was said to be efficient for updation of Weights**

**Advantages of Sigmoid Activation Function:**

(i) Smooth gradient, prevents jumps in output value

(ii) Output value between 0 to 1, Normalizing the output of each neuron

(iii) Clear Predictions, very close to 0 or 1

**3 Major Disadvantages:**

(i) Prone to Vanishing Gradient

(ii) Function output not zero centered

(iii) Power Operational is time consuming

**In Linear Regression, Standardization is helpful to make zero-centered and update the weights accordingly**

**10. Tanh Activation Function:** Hyperbolic Tangent Function; Function is Zero Centered

**Formula: tanh(x) = e^z - e^-z / e^z + e^-z**

**Values range between -1 to 1; Derivatives Range between 0 to 1**

**Advantages:** Zero Centered --> Better Weight Updation

**Disadvantages:**

1. Vanishing Gradient Problem still exists for Deep Layered Neural Networks (No Problem till Small to Medium Layered Neural Network)

2. Computational Time is more

**11. ReLu Activation Function:**

ReLu = Max(o,x)

<=o --> Output = 0

**Output of Derivative: 0 or 1** (whereas for Sigmoid 0 to 0.25; Tanh 0 to 1

Since output is 0 or 1, it solves Vanishing Gradient Problem

**If Derivative of ReLu Output is 1 --> Weight Updation will happen; If Zero --> Creates a Dead Neuron**

**When Derivative of ReLu(z) becomes 0? For Negative Values it becomes 0** [For Positive Values of (z) --> It becomes 1]

If 0 --> Dead Neuron; Other than that it solves the Vanishing Gradient Problem

**Advantages:**

(i) Solves Vanishing Gradient Problem

(ii) Max(0,x) --> Calculation is Super-fast; ReLu also has linear relationship

(iii) Much faster than Sigmoid/Tanh

**Disadvantages:**

(i) Dead Neuron

(ii) ReLu Output 0 (or) x; Not a Zero-Centric Output

**12. Leaky ReLu and Parametric ReLu:**

**ReLu stands for Rectified Linear Unit**

ReLu had Dead Neuron Problem, which leads to Dead ReLu Problem

Leaky ReLu and Parametric ReLu solves this problem

**Leaky ReLu: f(x) = max(0.01x,x) --> With respect to Forward Propagation**

**Derivative: 0.01 to 1 (Backward Propagation)**

**Parametric ReLu: max(αx,x)**

If 0.01 --> Leaky ReLu; If Alpha --> Parametric ReLu

**With respect to LeakyReLu, we don't get derivative as zero**

**Advantages:**

(i) LeakyReLu has all advantages of ReLu

(ii) It removes dead ReLu Problem

**Disadvantage:**

(i) Not Zero Centered

**13. Exponential Linear Unit:**

Used to solve ReLu problems

**f(x) = x, if x>0, α(e^x - 1), otherwise** [Forward Propagation]

If d(f(x) / dx = > 0, then 1 [Backward Propagation]

**Advantages:**

(i) No dead ReLu issues

(ii) Zero Centered

**Disadvantages:**

(i) Slightly more computationally intense

**14. Softmax Activation Function:**

**Softmax Activation function used in Multi-Class Classification problem**

Example for a Image Classification: In the output layerm we have some outputs of 4 Neurons with Values -1,0,3,5

**After that we use Softmax Activation Function to predict Multi-Class Classification**

**Softmax = e^yi / (Sum k=0,n e^yk)**

We calculate this for each output

**Finally, we divide each output by sum of all, to find which will be the final predicted output**

**Eg. after applying Softmax Formula, Cat - 0.0003, Dog - 0.0024, Monkey - 0.0183, Horse - 0.1353 **

**Final Prediction = Horse Value (Since Horse has higher value  here) / Sum of all from Cat to Horse = 86%**

**So, prediction will be Horse here**

**15. Which Activation Function when to apply?**

If Hidden Layer has Sigmoid, it leads to Vanishing Gradeint problem

**Hidden Layers --> Use ReLu or its variants such as Leaky ReLU, PreReLu, ELU**

**Output Layers --> Sigmoid (For Binary Classification); Softmax (For Multi-Class Classification)**

**16. Loss Function vs Cost Function:**

Loss Function: We pass each data point, we calculate y^, calculate loss and do backward propagation for every record to update weights (MSE= (y-y^)2

Cost Function: We update all data points at once, calculate (y-y^)2 for every points, do sum of all errors, find out mean and weight updation will be done at only once

**17. Regression Cost Function:**

ANN can be used for both Regression and Classification (Check Notebooks / Videos for Clear Formulae)

**Regression Cost Functions: Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss, Root Mean Squared Error (RMSE)**

**a) Mean Squared Error (MSE):**

**Loss Function: (y-yi)^2**

**Cost Function: 1/n (Sum i=1 to n (y-yi)^2**

**Advantages:**

(i) MSE is differentiable at every point

(ii) It has only one Global Minima

(iii) Convergence is faster

**Disadvantages:**

(i) Not robust to Outliers - It penalizes the outlier, means the best fit line gets moves towards the outlier direction

**b) Mean Absolute Error (MAE):**

**Loss Function: |y-y^|**

**Cost Function: 1/n Sum i=1 to n |y-y^|**

**Advantages:**

(i) Robust to Outliers (Best fit line will change, but not that much when compared to MSE)

**Disadvantages:**

(i) Convergence takes time (There is concepts of Sub-Gradients, i.e. we take separate sub gradients and calculate minima)

**If Outliers, use MAE; If no Outliers, use MSE (If we remove Outliers and use MSE, we may loose some data points**

**c) Huber Loss:**

Combination of MSE and MAE

Cost Function = 1/n (y-yi)^2, if |y-y^| <= δ

δ |y-y^| - 1/2 δ^2, Otherwise

δ - Hyperparameter - A threshold which we can think of

**d) Root Mean Squared Error (RMSE):**

Cost Function = √Sum i = 1 to n (y-y^)2 / N

**18. Which Loss Functions when to use?**

Hidden Layer --> Output Layer (Activation Function) --> Problem Statement ---> Loss Function

(i) ReLu or its variants - Sigmoid - Binary Classification - Binary Cross Entropy

(ii) ReLu or its variants - Softmax - Multi-Class Classification - Categorical or Sparse Categorical Cross Entropy

(iii) ReLu or its variants - Linear - Regression - MSE, MAE, Huber Loss, RMSE

**19. Types of Optimizers:**

(i) Gradient Descent

(ii) Stochastic Gradient Descent (SGD)

(iii) Mini Batch SGD

(iv) SGD with Momentum

(v) Adagrad and RMSPROP

(vi) Adam Optimizers

**20. Gradient Descent Optimizer:**

We use optimizers to reduce loss, with the help of Backward Propagation (Main Role of Optimizers: To update Weights)

**General Weight Updation Formula: Wnew = Wold - n(dL/dWold)**

**Our Main Goal in Gradient Descent: To reach Global Minima; When Global Minima is reached, Wnew = Wold and then no weight updation happens**

**MSE: Loss Function: (y-y^)2; Cost Function: 1/n Sum i=1 to n (y-y^)2**

**Two Important Concepts: Epoch, Iteration**

**Gradient Descent generally takes all data points at once, calculate loss and updates the weights back in Backward Propagation; This will happen in Multiple Epochs, as required according to the Problem Statement**

If Dataset has 1000 data points, we calculate y^ for all 1000 points, and then calculate cost function, and then weights will get updated accordingly for all points

**Epoch: 1 Forward Propagation with all data points + 1 Backward Propagation with weights getting updated**

**We do multiple epochs with the same steps, until the cost function gets reduced. As we do epochs, loss gets reduced**

**In this case, 1 Epoch = 1 Iteration**

**Iteration: If suppose I have 1000 data points, and I split it, and I send it as 1 iteration with 100 data points (As 10 iterations for 1000 points/10 Iterations, so 100 points for each iteraion)**

**1 Epoch can have any number of iterations and based on number of iterations we will have data  points**

**But in Gradient Descent, 1 Epoch = 1 Iteration, because it takes all data points at all once**

**Advantages:**

(i) Convergence will happen

**Disadvantages:**

(i) Huge Resources Needed: Say for 1M data points,huge RAM, GPU are needed. If I have 1M data points, we have to update that many of data points. To take up that many, store it, and then update, need more resources, including for the weight updation which needs GPU.

**If No Huge resources, system will get hung in Local. We can do in Cloud like AWS, but which is again cost based; Therefore overall it is Resource Intensive**

**21. Stochastic Gradient Descent (SGD):**

**In Stochastic Gradient Descent, we send only one data point in every iteration. Same way, one epoch can have any number of iterations**

**For Example, if 1000 datapoints, 1000 iterations will happen to complete one epoch. Iterations will go ahead until we reduce the loss value**

**Similarly, we can go ahead with 100 epochs too, until we reduce the loss/cost value**

**Advantages:**

(i) Solves resource intensive issue

**Disadvantages:**

(i) Time complexity --> Say if 1M records, for 100 epochs, it will run for 100 epoch x 1M record for each epoch.

(ii) Convergence takes more time.

(iii) Noise gets introduced.

**Earlier we saw that we took all data points, calculate y^, found out cost function, because of that Smoother convergence happened**

**But since we take only 1 data point, noise will be introduced,i.e., the point will roam around in Gradient Descent curve and then only comes to Global Minima. Because of this too, time complexity increases and convergence also takes more time.**

**22. Mini Batch with SGD:**

Along with Epoch, Iteration, will also learn about Batch-Size.

Let's say if 100k Data Points (100000), say batch size we say 1000 records, then number of records will be 100000/1000 = 100 Iterations

**For each iteration in an epoch we will send data points of 1000 (Like that 100 Iterations complete for 1 Epoch, contributing for Batch Size of 1000)**

Here, Let's say there is a 8GB RAM, which we used to send 1 record, it can handle 1000 records now. But handling 5000 records will be an issue, we may need 16GB RAM there

**Again because of this Noise will be there**

**In Mini-Batch SGD, Noise will be reduced because of batches, but still it will be there. We don't send 1 record, but we send in batches, where it takes a route to reach the Global Minima**

**Advantages:**

(i) Convergence speed will get increased

(ii) Noise will be less when compared to SGD

(iii) Efficient Resource utilization (e.g. RAM)

**Disadvantages:**

(i) Noise still exists - Efficient than SGD but still takes time to converge

**We will try to smoothen the cuvrve; Smootheing helps to reach the Global Minima faster; For that we use SGD with Momentum**

**23. SGD with Momentum:**

**Always with Weight Updation similar theory, we have to update Bias too, with the Formula of bnew = bold - n(dL/dbold)**

We know that weight updation formula is, Wnew = Wold - n(dL/dWold)

Here with momentum, we will take it as,

**Wt = Wt-1 - n (dL/dWt-1)** 

**We are trying to update current time, with respect to previous time**

**For Smoothening, we use Exponential Weighted Average**

**E.g. Value at Vt1=a1, then at Vt2 we calculate as**

**Vt2 = B x Vt1 + (1-B) x a2**

**Generally, we give as, Vt2 = 0.95 x Vt1 + 0.05 x a2**

**0.95, i.e. we give more importance to the previous value to control the smoothening and current point (say 0.05) which has lesser control over the smoothening. Same way third point, has similar second points control more**

**Vt3 = B x Vt2 + (1-B) x Vt3**

**We do exponential weighted average to smoothen the Curve**

**B (beta) Smoothening Parameter which controls smoothening; General Value 0 to 1**

**Advantages:**

(i) Reduces the Noise

(ii) Quicker Convergence

**Usually used in Time Series Problems with ARIMA, SARIMAX**
