# Introduction


For insurance companies it is of financial relevance to detect fraudulent claims as early as possible to be able to react accordingly (e.g. reject the claim). A common approach is often a simple fact check (e.g. if the historic weather data allow a weather-related claim), which is mostly executed by a claim adjuster. This can also be (partly) automatized by extracting information out of the claim-description document and other available data sources. However, chances are high that an automatized stand-alone fact check identifies a very limited number of potential fraud-cases. 
Moreover, insurance companies have historical knowledge of fraud cases in the past and it makes sense to use this common memory. This use-case describes how data science techniques can be used to achieve a better detection rate based on historic data. The idea is to train an **artificial neural network** (ANN) on the company's past data and use it after the learning process to identify new potential fraud cases.


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

# Metrics for Imbalanced Data
As mentioned before for the case of an imbalanced data set, the use of performance indicators has to be adapted accordingly. Let's now assume that a trained statistical model  *a<sup>L</sup>* leads for the claim case *x<sub>i</sub>* to a prediction in (0,1), e.g. gives a fraud probability, *a<sup>L</sup>(x<sub>i</sub>) ∈ (0,1)*.


A binary classification is then typically obtained by using a *threshold* *t<sub>hres</sub>* in the following manner: 

<p align="center">

<img width="370" height="115"  src="/ANN_images/Equations/ypred.png">
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

As for the activation function *f(.)*, there are different options to choose and whose performance heavily depends on the field of application. The most common three choices are: 

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

*X = (x<sub>1.</sub><sup>T</sup>, x<sub>2.</sub><sup>T</sup>, ..., x<sub>n.</sub><sup>T</sup>)<sup>T</sup> ∈ ℝ<sup>n x m*
 
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

Now, consider the case of initializing the parameters with *w = b = 2*. Here the output is with *a = σ(4) ≈ 0.98* even further away from the desired value *y = 0* and the error with *C<sub>MSE</sub> ≈ 0.96* even bigger than before. When visualizing the learning progress, a so-called **learning slowdown** has occurred: 
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

For a single output neuron with *a<sup>i</sup> = a<sup>L</sup>(x<sub>i.</sub><sup>T</sup>)* and *y<sup>i</sup> = y(x<sub>i.</sub><sup>T</sup>)* the cross-entropy is defined as:

<p align="center">
<img width="550" height="90"  src="/ANN_images/Equations/CE.png">
</p>

Performing the calculations just like before, following partial derivatives are obtained: 
<p align="center">
<img width="200" height="120"  src="/ANN_images/Equations/CE_wDerivative1n.png">
</p>

This is great news: now the rate of weight changes depends actually on the **error σ(z) – y**. This means the bigger the error is, the bigger the updates get! 

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
<img width="120" height="55"  src="/ANN_images/Equations/Update_Formula.png">
</p>

When using the sigmoid activation-function, the activation is non-negative, *a<sub>i</sub> = σ(z<sub>i</sub>) ≥ 0*, which implies that the sign of the gradient *dC/dw<sub>i</sub>* solely depends on *sgn(dC/dz<sup>^</sup>)* and is equal for *i = 1,2* and *3*. As a consequence, all three weight updates are either bigger or less than 0 and it’s impossible for one of these parameters to increase while a different one decreases. This however is clearly not the behavior we expect our model to possess since this could lead to a systematic bias along the weight updates. 

Even if it’s probably not necessary, we would at least want our model to have the ability to change a specific weight independently of another weight. 
Here, the hyperbolic tangent has clear advantages over the sigmoid function: Since the hyperbolic tangent is an odd function, *tanh(-z) = -tanh(z)*, and *a<sub>i</sub>* can be either negative or positive, we can expect the positive and negative activations to balance each other out and avoid any bias for the weight updates. Thus, usually the *tanh(.)* is a better option to choose as activation function than the sigmoid.

A similar argument could be made for a neural network where we are interested in the weight updates from the input to first hidden layer: 
<p align="center">
<img  width="370" height="220" src="/ANN_images/Scale_Motivation.png">
 </p>
 
As for the formula, the inputs x_i are plugged in instead of the activation function, so for *i = 1,2,3*:
<p align="center">
<img width="120" height="55"  src="/ANN_images/Equations/Scale_Formula.png">
</p>

To ensure no systematic bias for the updates, we settle for positive and negative inputs in the same extent. Therefore, it’s a common recommendation to **center** (and also **scale**) **the inputs** before training the model.  

#### Regularization Parameters

As standard neural networks contain non-linearities within their model structure (f.e. through activation functions), they are referred to as non-linear models and some proper technique needs to be established to deal with the danger of **overfitting**. 

Let’s for instance look at a popular example, the MNIST dataset for digit classification, publicly available at http://yann.lecun.com/exdb/mnist/. The dataset consists of handwritten digits, arranged in a 28 x 28 pixel array. That is, each digit is represented by a 784-dimensional vector, consisting of continuous greyscale-values between 0 and 1 for each pixel. Here are 4 exemplary digits:
<p align="center">
<img  width="360" height="360" src="/ANN_images/MNIST_example.png">
</p>

As for our neural network setup, we want to classify the handwritten digit once we obtain the 28 x 28 pixel-array. Therefore, we make use of the following structure [III]: 
-	784 neurons in input layer (28 x 28)
-	single hidden layer with 15 neurons
- output layer with 10 neurons, 1 neuron for each output, the prediction is obtained by taking the output neuron with the highest value (is most likely)

<p align="center">
<img  width="500" height="380" src="/ANN_images/MNIST_structure.png">
</p>

For demonstration purposes only 1.000 digits are chosen as training data and the performance is tested on 10.000 additional independent digits (test data). Training the neural network with a cross-entropy cost function led to the following history for each data part:

<p align="center">
<img  width="660" height="340" src="/ANN_images/MNIST_learnprocess1.png">
 </p>
 
When focusing on the later parts of the training process, here epoch 200 to 400, the phenomenon of overfitting is apparent: while the left plot indicates that the cost is decreasing and the model is getting better on the training data, the right plot shows how the accuracy hardly improves on the test data and behaves very volatile. 
 
Note: here we can expect the data to have no significant imbalance towards a certain digit, so the accuracy as metric is applicable.

<p align="center">
<img  width="660" height="340" src="/ANN_images/MNIST_learnprocess2.png">
 </p>

These two graphs emphasize the finding from before: the cost function on the test data achieves its minimum around epoch 15 (left plot). After that the cost function is constantly increasing and the model is getting worse on unseen data. Starting with epoch 15, the accuracy for the training data remains at 100 %. After the model learns to fit the parameters even more precise on the training data without leading to any improvement for the test data – so the model is overfitting. To cope with *overfitting*, two different methods are introduced: 

#### 1) Hold-Out
So how to prevent a neural network from overfitting? As seen in the example, it would be sufficient to stop the training process much earlier (around epoch 15), because after that the model gains no more improvements on unseen data and only fits more fine-grained on the training data. Thus, one natural approach to stop overfitting would be to stop the learning process once the cost doesn’t improve anymore on the test set. However, the test set is obviously needed for a final evaluation of the model and is not allowed to use in the training process, so we need a new set that is independent of the training set. 

Hence, we make use of a so-called **validation** or **development set**. Therefore, the full dataset X is split into 3 different parts. In our case we are doing this by using 4 equally-sized subsets and merging two of them into the training set: 
-	training set: first half of the data (part 1 & 2)
-	validation set: third part
-	test set: fourth part 

<p align="center">
<img  width="500" height="170" src="/ANN_images/holdout.png">
</p>

To ensure no systematic bias is present, it is recommended to shuffle the observations randomly before splitting them into the disjunct parts. 

The strategy of using a separate validation set to evaluate the cost on, is called **early stopping** or **hold-out** (since data is held out of the training process, instead used for evaluation). Different approaches exist for this method, for instance the “*no-improvement-in-ten-rule*”. This stops the learning process as soon as no improvements for the cost were obtained for the last 10 epochs on the validation data and then selects the parameters from the certain epoch where the cost-minimum was achieved. This “patience”-parameter of 10 is itself a hyperparameter of the model and can be varied accordingly to check how sensitive the model is to overfitting. 

The hold-out method is not specifically tied to neural networks, it can be applied for multiple models.

In general, the concept of overfitting is less of a problem for bigger data sets. In the example before however the size of the training set was intentionally chosen very small with 1.000 observations - this made the problem of overfitting more apparent.

#### 1) Dropout

Another method that is commonly used for dealing with overfitting problems is the concept of an ensemble learner with a subsequent majority vote (like applied in random forests). These ensembles are very effective, especially if some kind of randomness is involved in the machine learning model and slight adjustments within the data lead to (entirely) different models. For neural networks, this ensemble learner can be simulated with the so-called **dropout** method. The idea is to train multiple neural networks that are slightly different in their structure and obtain the final predictions by averaging the individual predictions for all the sub-models. 

To achieve “slightly different” models, each hidden-neuron is temporarily removed with a probability of *p* (again a hyperparameter) within each mini-batch and the parameter updates are performed on the remaining neurons. Because of this dropout, the weights are actually larger for the remaining neurons, and so as compensation the weights in the corresponding layer are scaled by *(1-p)* for the final predictions. 

Here is a graphical representation of two different mini-batches when dropout is applied with *p = 0.5* to the single hidden layer: 

<p align="center">
<img  width="650" height="260" src="/ANN_images/Dropout_batch1.png">
</p>

<p align="center">
<img  width="650" height="260" src="/ANN_images/Dropout_batch2.png">
</p>

That this dropout-method performs well, can be seen here for the aforementioned example of the MNIST digit classification: 
<p align="center">
<img  width="660" height="340" src="/ANN_images/MNIST_learnprocess_dropout.png">
 </p>
 
Not only is the accuracy on a higher level with dropout, but also the history shows an improved smooth development and increase in accuracy for the later epochs.

# Neural Networks - Use Case
After going into detail on a few theoretical aspects of neural networks, we will now focus on the practical side by illustrating the approach on the specific use-case. As for the implementation, the main-package used in **R** was **keras** [7].

## Data preparation
After gathering the necessary claim data, the first step is to look for any anomalies. This includes searching for obvious outliers and imputing missing values, since R can’t handle missing values in the keras-package. Here, **hot-deck imputation** was carried out. See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3130338/ for more information.

After that, the numerical variables were centralized and scaled, and the categorical variables encoded via dummy encoding [8]. To wrap things up, the data was shuffled randomly and put into four equal parts. For imbalanced data it is recommended to apply **stratified sampling** to ensure a similar target distribution in each data part [9].

## Neural Network Settings
The ultimate goal of our analysis was to tune the neural network as fine-grained as possible in order to achieve a high performance. Thus, multiple different approaches were applied and compared, like 
-	varying the number of hidden layers
-	try additional transformations (like PCA)
-	make use of over- and under-sampling to cope with the presence of unbalanced data [10] 

However, the crucial part for tuning neural networks lies in the high-dimensional space of possible hyperparameters. There doesn’t exist a clear-cut theory that advises the user to choose certain hyperparameters for specific tasks, it is rather a **trial-and-error** approach: try out different values and choose what works best. 

Especially in neural networks, lots of different hyperparameters arise, such as: 

-	activation function per layer
-	number of hidden neurons per layer
-	number of hidden layers
-	dropout rate per layer
-	rate for under/over-sampling 
-	patience-parameter for early stopping
-	learning rate eta for stochastic gradient descent
-	mini-batch size B
-	etc.

This clearly represents the bottleneck when working with neural networks.

In general, it is advised to carry out the **hyperparameter tuning** by making use of the validation data: Define different settings, train the model with the corresponding hyperparameters and evaluate the performance on the validation data. Once reasonable values are found for a certain hyperparameter, go onto the next hyperparameter and iterate the process. The most important hyperparameters like the learning rate should then be backtested a few times.

The concept of tuning the hyperparameters is by far the most important aspect of training a good neural network. To get a more detailed understanding, we would recommend two sources: 
-	https://towardsdatascience.com/hyperparameters-optimization-526348bb8e2d
-	https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594

But even with the optimizing of the hyperparameters, the results still could heavily depend on how we chose to split our data in the first place. A more robust method is to use **k-fold Cross-Validation** [11], in our case with *k = 4*. With this method the model is not trained only once with a single training, validation and test set, but three additional times with different splits corresponding to different data sets: 

<p align="center">
<img  width="650" height="220" src="/ANN_images/4foldCV.PNG">
</p>

This means that the hyperparameter tuning becomes even more expensive as we would have to perform it four different times for each training and validation set. In this use-case, the optimization was restricted to the first split and once the top-n settings were found (“candidates”), these n different combinations of hyperparameters were tested for the other splits in order to check which led to the best performance. 

## Results
One major advantage of applying cross-validation is that predictions are obtained for all four disjoint test sets. Thus, we ended up with a prediction for **all claims** of the data set and could asses the performance on the whole data set. 

As for the tradeoff between precision and recall, the **PR-curve** of our predictions indicates that the model performs very well in terms of precision for a recall below 0.52. Once a higher recall is desired, the precision drops off significantly. We assume that this is because at this point the by far most important variable lost its predictive power. A similar trend was discovered when comparing neural networks with other Machine Learning models that are indeed explainable.

<p align="center">
<img  width="620" height="360" src="/ANN_images/Result_PR.png">
</p>

The shape of the **MCC-curve** proves that the predictions are stable when settling for a threshold in [0.2,0.8]. The highest MCC-value was achieved for a threshold about 0.38 (black dot), i.e. the model only predicts a fraud case when the predicted probability exceeds 0.38. 
When we are interested in a big number of potential fraud cases, which means we accept a low threshold, the performance drops off significantly. A similar phenomenon was seen in the PR-curve above. 

<p align="center">
<img  width="600" height="360" src="/ANN_images/Result_MCC.PNG">
</p>


# Sources

As a main source, the following book was used:

[1] 	http://neuralnetworksanddeeplearning.com/ 

Most of the concepts presented in this document, can be found there.



Other sources, that might give the reader additional insights: 

[2]	https://en.wikipedia.org/wiki/Confusion_matrix

[3]	https://en.wikipedia.org/wiki/Precision_and_recall

[4] 	https://en.wikipedia.org/wiki/F1_score

[5]	https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

[6] 	https://en.wikipedia.org/wiki/Cross_entropy

[7] 	https://cran.r-project.org/web/packages/keras/keras.pdf

[8]	https://en.wikipedia.org/wiki/Dummy_variable_(statistics)

[9] 	https://en.wikipedia.org/wiki/Stratified_sampling

[10]	https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis

[11]	https://en.wikipedia.org/wiki/Cross-validation_(statistics)


Images:

[I] 	https://newvitruvian.com/explore/crash-clipart-cartoon-car/#gal_post_1649_accident-clipart-comic-13.jpg [accessed May 22, 2019]
https://medium.com/intro-to-artificial-intelligence/deep-learning-series-1-intro-to-deep-learning-abb1780ee20 [accessed May 22, 2019]

[II] 	lecture Machine Learning, TU Vienna, WS2019	

[III] 	https://www.researchgate.net/figure/Neural-network-with-hidden-layer-for-MNIST-data_fig2_308120458 [accessed Apr 18, 2019]
