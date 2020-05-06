# Introduction


For insurance companies it is of financial relevance to detect fraudulent claims as early as possible to be able to react accordingly (e.g. reject the claim). A common approach is often a simple fact check (e.g. if the historic weather data allow a weather-related claim), which is often executed by a claim adjuster. This can also be (partly) automatized by extracting information out of the claim-description document and other available data sources. However, chances are high that an automatized stand-alone fact check identifies a very limited number of potential fraud-cases.
Moreover, insurance companies have historical knowledge of fraud cases in the past and it makes sense to use this common memory. This Use Case describes how data science techniques can be used to archive a better detection rate based on historic data. The idea is to train an artificial neural network (ANN) on the company's past data and use it after the learning process to identify new potential fraud cases.


# Data Setup


## Prerequisites
Every data base will have different data items which can be used in setting up an ANN. But a field indicating if a historic claim-record has been identified to be fraud is mandatory. This yes/no field is essential for training the ANN and must be added if not available (which is called "data labelling"). The boolean variable is also the outcome of the model.

## Setup of Use-Case
In our case we will demonstrate the theoretical concepts by applying it for a concrete use case from UNIQA. The underlying data is based on casco-car insurance claims over a observation period of 3 1/2 years. After gathering the corresponding data, we ended up with about 60 explanatory variables and north of 300.000 claims. 
The sheer size of the data might excite the reader at first glance because it's a well known fact thatespecially neural networks gain huge performance boosts when applying bigger datasets (**Big Data**), but really all we care about is the distribution of the variable of intereset - in our case a binary indicator for fraud. We observed a event rate of about 0.44%, so not even every 200th claim case was detected as a fraudulent one. Thus, the distribution of our target variable is heavily skewed towards the majority class and our underlying data falls under the category of an **imbalanced data set**. This fact has to be considered twice when training a statistical model: 
 1. What algorithm should we choose for our model? Is the model able to handle imbalanced data? Should we apply some changes to the algorithm? Per default, most algorithms intend to maximize the overall accucary/error and don't focus on the minority class.
 2. How to assess the model performance since the target class is skewed? Should other performance indicators than the traditional ones should be considered? 
 
The answer to the second question will be discussed in the next section. As for the first question, we will come back to that point when we talk about our model structure.

As for our use-case, it might be beneficial to first look at the *basic setup*. Once we finished training our neural network on the previously-extracted historical claim data, the goal is to apply the model every time a new claim is planted. In this case all the explanatory variables are available and we can obtain a prediction if the corresponding claim is fraudulent by applying the data to our trained model: 

![](/ANN_images/Motivation_Setup.png)

Through the model we will either receive a binary prediction variable with the value 1 indicating the the claim is a fraud or - what might be ever better - a fraud probability. One might prefer the use of probabilities, since this allows to actually detect how sure the model is in it's prediction to be a fraud case. 

Given a set of *n* observations (**x**<sub>1</sub>, **x**<sub>2</sub>, ..., **x**<sub>*n*</sub>) with **x**<sub>*i*</sub> ∈ ℝ<sup>*k*</sup>, the *k*-means algorithm partitions the *n* observations into *k* ≤ *n* sets *C* = {*C*<sub>1</sub>, *C*<sub>2</sub>, ..., *C*<sub>*k*</sub>} such that

# Metrics for Imbalanced Data
As mentioned before, for the case of a imbalanced data set the use of performance indicators has to be adapted accordingly. Let's now assume that a trained statistical model *a<sup>L</sup>* leads for the claim case *x<sub>i</sub>* to a prediction in (0,1), e.g. gives a fraud probability, 

                                                 *a<sup>L</sup>(x<sub>i</sub>) ∈ (0,1)*



A binary classification is then typically obtained by using a *threshold* <img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\large&space;t_{hres}" title="\large t_{hres}" /> in the following form, 

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\large&space;y&space;=&space;\left\{&space;\begin{array}{ll}&space;0,&space;&&space;a^L(x_i)&space;\,\leq&space;\,t_{hres}\\&space;1,&space;&&space;a^L(x_i)&space;\,\,\,&space;\textgreater&space;\,\,&space;t_{hres}&space;\end{array}&space;\right.&space;\label{2_1}" title="\large y = \left\{ \begin{array}{ll} 0, & a^L(x_i) \,\leq \,t_{hres}\\ 1, & a^L(x_i) \,\,\, \textgreater \,\, t_{hres} \end{array} \right. \label{2_1}" />
 </p>

e.g. if the probability exceeds the threshold, the claim is classified as a fraud, otherwise as non-fraud. This method allows to get *binary predicitons* from continuous values like scores or probabilities. For this binary values then the common **confusion matrix** can be visualized: 


<p align="center">
  <img height=350 width=400  src="/ANN_images/Confusion_Matrix.png">
</p>

Basically, the two error types that can occur, are placed in the off-diagonal of the confusion matrix. These errors are familiar from hypothesis testing:
- **FP**...Type 1 Error (false positive): a non-fraud claim is classified as fraud
- **FN**...Type 2 Error (false negative): a fraud claim is not classified as fraud




# Design of ANN
TODO

# Performance
The quality of Fraud modelling can be determined by looking at certain indicators which meassure the success rates. Basically, two error types can occur, which are familiar from hypothesis testing:
- Type 1 Error (false positive): a non-fraud claim is classified as fraud
- Type 2 Error (false negative): a fraud claim is not classified as fraud

The accuracy rate in fraud detection is the percentage of the true positive in relation to all positive fraud cases (i.e. the sum of true positives and false negative).

# Conclusion
**Pros:** ANNs are powerful at finding non-linear and very complex relations in large datasets. They accuracy is usually very high.

**Cons:** The interpretability is very limited and decision cannot easily be reasoned out ("black box").
