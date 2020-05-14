# Introduction


For insurance companies it is of financial relevance to detect fraudulent claims as early as possible to be able to react accordingly (e.g. reject the claim). A common approach is often a simple fact check (e.g. if the historic weather data allow a weather-related claim), which is mostly executed by a claim adjuster. This can also be (partly) automatized by extracting information out of the claim-description document and other available data sources. However, chances are high that an automatized stand-alone fact check identifies a very limited number of potential fraud-cases. 
Moreover, insurance companies have historical knowledge of fraud cases in the past and it makes sense to use this common memory. This use-case describes how data science techniques can be used to achieve a better detection rate based on historic data. The idea is to train an artificial neural network (ANN) on the company's past data and use it after the learning process to identify new potential fraud cases.


# Data Setup


## Prerequisites
Every data base will have different data items which can be used for setting up an ANN, but a field indicating if a historic claim-record has been identified to be fraud is mandatory. This yes/no field is essential for training the ANN and must be added if not available (this is called "data labelling"). The boolean variable is also the outcome of the model.

## Setup of Use-Case
In our case we will demonstrate the theoretical concepts of neural networks by applying them on a concrete use-case from UNIQA. The underlying data is based on Motor Hull-car insurance claims over an observation period of 3,5 years. After gathering the corresponding data, we ended up with about 60 explanatory variables and north of 300.000 claims. 
The sheer size of the data might excite the reader at first glance since it is well known that especially neural networks gain huge performance boosts when applying bigger datasets (**Big Data**). But really all we care about is the distribution of the variable of interest - in our case a binary indicator of fraud. We observed an event rate of less than 1%, so only very few claim cases were marked as fraudulent ones. Thus, the distribution of our target variable is heavily skewed towards the majority class and our underlying data falls under the category of an n **imbalanced data set**. This fact must be considered twice when training a statistical model:

 1. What model should we choose (which algorithm should be selected)? 
The model should be able to handle imbalanced data. Per default, most algorithms intend to maximize the overall accuracy and don't focus on the minority class.

 
 2. How to assess the model performance since the target class is skewed? Maybe some other performance indicators than the traditional ones should be considered.
 
The answer to the second question will be discussed in the next section. As for the first question, we will come back to that later when we talk about our model structure.

It might be beneficial to first look at the basic setup for our use-case. Once we finished training our neural network on the previously extracted historical claim data, the goal is to apply the model every time a new claim is planted. In this case all the explanatory variables are available, and we can obtain a prediction whether the corresponding claim is fraudulent by applying the data to our trained model [I]: 

<p align="center">
 <img   src="/ANN_images/Motivation_Setup.png">
</p>

Through the model we will either receive a binary prediction variable with the value 1 indicating that the claim is a fraud or - which might be ever better - a **fraud probability**. The display of probabilities gives an impression of how accurate the model think it is in its prediction of a fraud case. 

Given a set of *n* observations (**x**<sub>1</sub>, **x**<sub>2</sub>, ..., **x**<sub>*n*</sub>) with **x**<sub>*i*</sub> ∈ ℝ<sup>*k*</sup>, the *k*-means algorithm partitions the *n* observations into *k* ≤ *n* sets *C* = {*C*<sub>1</sub>, *C*<sub>2</sub>, ..., *C*<sub>*k*</sub>} such that

# Metrics for Imbalanced Data
As mentioned before for the case of an imbalanced data set, the use of performance indicators has to be adapted accordingly. Let's now assume that a trained statistical model  *a<sup>L</sup>* leads for the claim case *x<sub>i</sub>* to a prediction in (0,1), e.g. gives a fraud probability, *a<sup>L</sup>(x<sub>i</sub>) ∈ (0,1)*.


A binary classification is then typically obtained by using a *threshold* *t<sub>hres</sub>* in the following manner: 

<p align="center">

<img width="330" height="115"  src="/ANN_images/Equations/ypred.png">
</p>

If the probability exceeds the threshold, the claim is classified as a fraud, otherwise as non-fraud. This method allows to get *binary predictions* from continuous values like scores or probabilities. For these binary values the common **confusion matrix** can be visualized [2]:


<p align="center">
  <img width="370" height="330"  src="/ANN_images/Confusion_Matrix.png">
</p>

Basically, the two error types that can occur are placed in the off-diagonal of the confusion matrix. These errors are familiar from hypothesis testing:
- **FP**...type 1 error (false positive): a non-fraud claim is classified as fraud
- **FN**...type 2 error (false negative): a fraud claim is not classified as fraud

Note: from here on, we will denote the **minority class** (class of interest – in our case the fraud claims) with **1** and the majority class with 0. 

Traditionally, the **accuracy**-metric is used for measuring the performance of a model. That is, divide all correctly classified observations by the total number. In terms of elements of the confusion matrix, this corresponds to:

<p align="center">

<img width="290" height="60"  src="/ANN_images/Equations/acc.png">
</p>

The higher the accuracy the better the model, with 1 being the maximum. 

However, the drawback of using the accuracy without inspecting the data is apparent: 
-	Consider our dataset for detecting claims and assume a distribution ratio of 99: 1
-	Let’s predict each single claim as non-fraudulent 
-	This would imply TN = 0.99*n and TP= 0 and in thus an accuracy of 0.99: 
-	The accuracy is very close to 1 and would indicate a nearly perfect model but would not detect a single fraud case correctly. 
-	This makes it obvious: for dealing with imbalanced data we need to define other metrics that can handle the fact that we are only  	    interested in recognizing the elements of the minority class (TP) as correctly as possible, while the true negatives are only of minor interest.

Two common measures to deal with class imbalance are **recall** (= sensitivity) and **precision** [3]:

<p align="center">
<img width="250" height="110"  src="/ANN_images/Equations/precrec.png">
</p>

Recall describes how many of all fraud cases (TP + FN) are correctly detected by the model (TP). Precision indicates how many of the predicted fraud cases (FP + TP) are indeed fraudulent (TP). 
To recap: recall deals with the question “how many?” and precision with “how exact?”. 

These two metrics are typically inversely related, so when recall increases the precision decreases. Furthermore, recall and precision should always be reported together, since we could for instance easily optimize recall independently to be 1 when only predicting fraud cases with our model (FN=0). In this case the precision would be close to 0. 

As stated before, the confusion matrix stems from assuming a fixed threshold value to create binary predictions. Varying this threshold leads to different confusion matrix values and therefore to different precision and recall values. Thus, the metrics described here are actually functions of the threshold, *rec = rec(t<sub>hres</sub>)* and *prec = prec(t<sub>hres</sub>)*.

Furthermore, we can easily plot the recall (x-axis) against the precision (y-axis) as a function of the threshold. This **precision-recall-plot** is commonly used especially for cases of imbalanced datasets and we’ll come back to this once we evaluate our final model performance. 

Since recall and precision should only be considered together and the use of two different metrics is tedious, it would be nice to come up with a single scalar value that somehow combines the two aspects. Hence, the **F1-score** is defined as the harmonic mean of precision and recall [4]: 

<p align="center">
<img width="560" height="75"  src="/ANN_images/Equations/F1.png">
</p>

Like for the accuracy, a value closer to one indicates a better model. 

While the F1-score is a very convenient metric to measure model performance, it comes with a slight disadvantage: it doesn’t consider TN at all. For a practical example as to why this may lead to problems, see for instance https://en.wikipedia.org/wiki/Matthews_correlation_coefficient. 

An alternative and better option is the so-called **Matthews Correlation Coefficient (MCC)**, which is regarded as being one of the best measures for describing the confusion matrix with a single value by many authors. It is calculated via: 

<p align="center">
<img width="560" height="75"  src="/ANN_images/Equations/MCC.png">
</p>

The fact that it is indeed a correlation coefficient – namely between the ground truth and the prediction – makes it easy to interpret. MCC= 0 indicates that the prediction is uncorrelated with the true class membership. 
For more detailed information, see the aforementioned Wikipedia-page [5]. 

Once the fixed threshold is dropped and different thresholds are considered, the MCC becomes a function, *MCC(t<sub>hres</sub>)*, and can be easily visualized against the threshold. We will see an example of this MCC-curve later on when discussing the model performance for our use-case.

# Neural Networks - Theory

Before going into details for our particular use-case, we want to tackle a few important aspects of modelling with feedforward neural networks (**FNN**). Instead of introducing neural networks step-by-step here, we assume that some knowledge about the general structure and functionality of neural networks is already present. If not – or if someone is interested in a quick refresher – these sources are a nice starting point: 

1)	Blog on https://towardsdatascience.com/@tushar20
-	Idea on Deep Learning: https://towardsdatascience.com/deep-learning-d5fe55326e57
-	Structure of FNN + learning parameters: https://towardsdatascience.com/deep-learning-feedforward-neural-network-26a6705dbdc7
-	Backpropagation-Algorithm: https://towardsdatascience.com/back-propagation-414ec0043d7

2)	Informative Video-Series on https://www.youtube.com/
-	Part 1) Deep Learning + Structure : https://www.youtube.com/watch?v=aircAruvnKk
-	Part 2) Gradient Descent + learning parameters: https://www.youtube.com/watch?v=IHZwWFHWa-w
-	Backpropagation-Algorithm: https://www.youtube.com/watch?v=Ilg3gGewQ5U
-	Mathematical details for Backpropagation: https://www.youtube.com/watch?v=tIeHLnjs5U8

## Hyperparameters 
A neural network is primarily defined via its **hyperparameters**, i.e. parameters that are not learned in a learning process but must be defined a-priori in order to obtain a certain structure. Consider the following architecture of a neural network [II]: 

<p align="center">
<img  width="650" height="350" src="/ANN_images/architecture.png">
</p>

When assuming m neurons in the input layer and k neurons in the subsequent first hidden layer, the following notation holds:
- *x ∈ ℝ<sup>m*... single observation = claim case
- *w ∈ ℝ<sup>k x m*... weight matrix between input and first hidden layer
- *b<sup>1</sup> ∈ ℝ<sup>k*... bias terms of first hidden layer
- *z<sup>1</sup> ∈ ℝ<sup>k*... weighted input of first hidden layer, that is: *z<sup>1</sup> = wx + b<sup>1</sup>*
- *a<sup>1</sup> ∈ ℝ<sup>k*... activation of first hidden layer
- *L*... number of layers, so *a<sup>L</sup>* is the final ouput of the model
 
The activation is calculated by applying some sort of activation function *f(.)* on the weighted inputs: 
<p align="center">
<img width="250" height="30"  src="/ANN_images/Equations/activation_eq.png">
</p>

*w* and *b* are the actual (trainable) variables of the model and will be trained accordingly through the learning process, typically via the **Backpropagation algorithm**. 

### Possible Activation Functions

As for the activation function f, there are different options to choose and whose performance heavily depends on the field of application. The most common three choices are: 

#### 1) Sigmoid function
The sigmoid function is also known as logistic function and maps the weighted inputs from the real axis onto the interval (0,1): 
<p align="center">
<img width="260" height="60"  src="/ANN_images/Equations/sigmoid_eq.png">
</p>

<p align="center">
<img  width="510" height="270" src="/ANN_images/sigmoid.png">
</p>

Higher weighted inputs obtain higher activations and vice versa. 
The sigmoid is commonly used as default for basic neural network applications.

#### 2) Hyperbolic tangent
When choosing the hyperbolic tangent instead of sigma(.) as activation function, the activations may also receive negative values: 
<p align="center">
<img width="330" height="60"  src="/ANN_images/Equations/tanh_eq.png">
</p>

Furthermore, the hyperbolic tangent is symmetric around the origin: 
<p align="center">
<img  width="540" height="280" src="/ANN_images/tanh.PNG">
</p>

As one could expect from the shape, *tanh(.)* is basically just a scaled version of *σ(.)*: 

<p align="center">
<img width="480" height="130"  src="/ANN_images/Equations/tanh_sigmoid.PNG">
</p>

#### 3) Rectified Linear Units
A common choice especially for problems regarding image recognition, are the so called rectified linear units (**relu**). This function maps the weighted inputs from the real axis onto the positive real axis united with zero, which ensures – similar to *σ(.)* – non-negative activations: 

<p align="center">
<img width="330" height="60"  src="/ANN_images/Equations/relu_eq.png">
</p>

<p align="center">
<img  width="540" height="280" src="/ANN_images/relu.PNG">
</p>


### Possible Cost Functions
One essential part of building statistical models is the ability to quantify the performance of the model based on the data, typically the training data. In our use-case setting we are dealing with the task of **supervised learning**, this means we can directly compare the variable of interest with the outcome of the model. Let’s assume our training set consists of n claim cases: 

*X = (x<sub>1.</sub><sup>T</sup>, x<sub>2.</sub><sup>T</sup>, ..., x<sub>n.</sub><sup>T</sup>) ∈ ℝ<sup>n x m*
 
 with *x<sub>i.</sub><sup>T</sup> ∈ ℝ<sup>1 x m* denoting the i-th observation as a row.

Once the model provides - based on some parameters *w* and *b* - an outcome *a<sup>L</sup>(x<sub>i.</sub><sup>T</sup>)* of the neural network, this value can be compared to the desired output *y(x<sub>i.</sub><sup>T</sup>)* of the training data.

Naturally, we would want these values to be as close as possible to each other and a way to measure this is by using the L2-norm and summing over all n observations in the training set. This leads to the common **MSE/L2 – cost function**: 

<p align="center">

<img width="500" height="90"  src="/ANN_images/Equations/MSE.png">
</p>

*C<sub>MSE</sub>* is non-negative and a smaller value would indicate a better model. Thus, in order to find good parameters w and b of the model, the cost function is usually numerically minimized via (stochastic) gradient descent. This approach forms the basis of the popular Backpropagation algorithm. However, there also some drawbacks when using the MSE- cost function, as for instance in the following setting.

**Example:** Consider the trivial case of a neural network that receives a single input 1 and should always provide the output 0. We use only a single training example and a sigmoid activation function. 

<p align="center">
<img  width="250" height="100" src="/ANN_images/Trivial_Example.png">
</p>

In the first step, let’s initialize the parameters with *w = 0.6* and *b = 0.9*. Executing the model with these initial values results in an output of *a = σ(z) = σ(1.5) ≈ 0.82*.
Hence, the discrepancy between a and y is big, therefore *C<sub>MSE</sub> ≈ 0.68* and the parameters must be adapted quite a lot in order to decrease the error properly. Training the model under this setting results in the following training history: 

<p align="center">
<img  width="340" height="310" src="/ANN_images/MSE_ex1.png">
 </p>
Rapid improvements in terms of costs are gained within the first few epochs, so the model seems to be recognizing early on in which direction the parameters w and b should be adapted. 

Now, consider the case of initializing the parameters with *w = b = 2*. Here the output is with *a = σ(4) ≈ 0.98* even further away from the desired value $y = 0$ and the error with *C<sub>MSE</sub> ≈ 0.96* even bigger than before. When visualizing the learning progress, a so-called **learning slowdown** has occurred: 
<p align="center">
<img  width="340" height="310" src="/ANN_images/MSE_ex2.png">
 </p>
 
Caused by the large activation and cost function, the model takes about 80 epochs to even start learning the parameters properly. From the start up to epoch 80, a lot of time has passed but hardly any improvements were made in the model. What’s the reason for the learning slowdown and how can we avoid this from happening? 

Let’s take a closer look at the cost function for this simple setting of a single training example (*n = 1*) and calculate the partial derivative of *C<sub>MSE</sub>=(y - a)<sup>2</sup>*: 

<p align="center">
<img width="650" height="65"  src="/ANN_images/Equations/MSE_wDerivative_1.png">
</p>

The chain rules and the following identities were used:
- *x = 1*
- *y = 0*
- *z = wx + b*
- *a = σ(z) = σ(wx + b)*

This ultimately results in: 
<p align="center">
<img width="300" height="60"  src="/ANN_images/Equations/MSE_wDerivative_2.png">
</p>

As stated before, neural networks are typically trained according to gradient descent and the adaption of the parameters is determined by the gradient. Here, the partial derivatives with respect to *w* and *b* both depend on the derivative of the sigmoid function in *z*. When plugging in the concrete values of our example, we can see that the potential changes of the parameters are of much bigger magnitude for the first case with the smaller error at the start: 

<p align="center">
<img width="260" height="110"  src="/ANN_images/Equations/MSE_wDerivative_3.png">
</p>

As the shape of *σ'(.)*
<p align="center">
<img  width="540" height="290" src="/ANN_images/sigmoid_der.png">
</p>
indicates, the parameter updates can be much bigger for weighted inputs around zero and the bigger the value of |*z*| becomes. The smaller the updates get the more time the neural network needs to find proper parameter values.

This is one of the main drawbacks when using the MSE-cost function and in general it would be better to decide for an alternative option, such as the **cross-entropy cost function**. This cost function is based on the concept of the Kullback-Leibler divergence to compare two probability distributions – for more details see [6]. 

For a single output neuron with *a<sup>i</sup> = a<sup>L</sup>(x<sub>i.</sub><sup>T</sup>)* and *y<sup>i</sup> = y(x<sub>i.</sub><sup>T</sup>)* the cross-entropy with is defined as:

<p align="center">
<img width="550" height="90"  src="/ANN_images/Equations/CE.png">
</p>

Performing the calculations just like before, following partial derivatives are obtained: 
<p align="center">
<img width="200" height="120"  src="/ANN_images/Equations/CE_wDerivative1n.png">
</p>

This is great news: now the rate of weight changes depends actually on the **error s(z) – y**. This means the bigger the error is, the bigger the updates get! 

Visualizing the training history with the cross-entropy cost function leads to:
<p align="center">
<img  width="660" height="340" src="/ANN_images/CE_ex1and2.png">
 </p>

with the values of the initial cost-function being *1.72* for (1) and *4* for (2). Not only the error is bigger in the second case, but when using the *C<sub>CE</sub>* also the initial update: 
<p align="center">
<img width="400" height="100"  src="/ANN_images/Equations/CE_wDerivative2.png">
</p>

This is how we would expect a cost function to behave. 

### Selection of Hyperparameters

#### Cost Function
As explained in previous chapter the learning potential is much higher for the cross-entropy cost function and therefore this cost function was selected. In our use-case of fraud detection the utilization of the cross-entropy CF led to major improvements compared to the MSE-based metric

#### Activation Function
Suppose the following neural network,
<p align="center">
<img  width="550" height="220" src="/ANN_images/Update_Motivation.png">
 </p>
 
and we are only interested in the connection between the activations *a<sub>1</sub>*, *a<sub>2</sub>* and *a<sub>3</sub>* to the weighted input *z<sup>^</sup>* in the next layer. 
Without going into any details: the backpropagation algorithm gives us the following relation for the gradient of the cost function with respect to a specific weight for *i = 1,2,3*:
<p align="center">
<img width="130" height="60"  src="/ANN_images/Equations/Update_Formula.png">
</p>

When using the sigmoid activation-function, the activation is non-negative, *a<sub>i</sub> = σ(z<sub>i</sub>) ≥ 0*, which implies that the sign of the gradient *dC/dw<sub>i</sub>* solely depends on *sgn(dC/dz<sup>^</sup>)* and is equal for *i = 1,2* and *3*. As a consequence, all three weight updates are either bigger or less than 0 and it’s impossible for one of these parameters to increase while a different one decreases. This however is clearly not the behavior we expect our model to possess since this could lead to a systematic bias along the weight updates. 

Even if it’s probably not necessary, we would at least want our model to have the ability to change a specific weight independently of another weight. 
Here, the hyperbolic tangent has clear advantages over the sigmoid function: Since the hyperbolic tangent is an odd function, *tanh(-z) = -tanh(z)*, and *a<sub>i</sub>* can be either negative or positive, we can expect the positive and negative activations to balance each other out and avoid any bias for the weight updates. Thus, usually the tanh(.) is a better option to choose as activation function than the sigmoid.


