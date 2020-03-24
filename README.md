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

![](ANN-images/Motivation_Setup.jpg)



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
