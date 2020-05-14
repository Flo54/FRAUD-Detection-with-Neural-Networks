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
