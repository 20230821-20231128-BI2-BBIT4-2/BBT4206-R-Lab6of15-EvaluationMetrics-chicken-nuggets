Business Intelligence Project
================
\<Group - Chicken Nuggets\>
\<24/10/2023\>

- [Student Details](#student-details)
- [ACCURACY AND COHEN’S KAPPA](#accuracy-and-cohens-kappa)

# Student Details

|                                              |                    |
|----------------------------------------------|--------------------|
| **Student ID Number**                        | 133824             |
| **Student Name**                             | Konse Habiba Siba  |
| **BBIT 4.2 Group**                           | 4C                 |
| **BI Project Group Name/ID (if applicable)** | Chicken-Nuggets    |
|                                              |                    |
| ———————————————-                             | ——————–            |
| **Student ID Number**                        | 137118             |
| **Student Name**                             | Fatoumata Camara   |
| **BBIT 4.2 Group**                           | 4C                 |
| **BI Project Group Name/ID (if applicable)** | Chicken-Nuggets    |
| ———————————————-                             | ——————–            |
| **Student ID Number**                        | 127039             |
| **Student Name**                             | Ayan Ahmed         |
| **BBIT 4.2 Group**                           | 4C                 |
| **BI Project Group Name/ID (if applicable)** | Chicken-Nuggets    |
| ———————————————-                             | ——————–            |
| **Student ID Number**                        | 136869             |
| **Student Name**                             | Birkanwhal Bhambra |
| **BBIT 4.2 Group**                           | 4C                 |
| **BI Project Group Name/ID (if applicable)** | Chicken-Nuggets    |
| ———————————————-                             | ——————–            |
| **Student ID Number**                        | 127602             |
| **Student Name**                             | Trevor Anjere      |
| **BBIT 4.2 Group**                           | 4C                 |
| **BI Project Group Name/ID (if applicable)** | Chicken-Nuggets    |

\#Loading packages

``` r
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: ggplot2

``` r
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: caret

    ## Loading required package: lattice

``` r
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: mlbench

``` r
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: pROC

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}
```

    ## Loading required package: dplyr

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

# ACCURACY AND COHEN’S KAPPA

``` r
# Loading the dataset
library(readr)
Customer_Churn <- read_csv("data/Customer Churn.csv", 
    col_types = cols(Complains = col_factor(levels = c("1", 
        "0")), `Tariff Plan` = col_factor(levels = c("1", 
        "2")), Status = col_factor(levels = c("1", 
        "2")), Churn = col_factor(levels = c("1", 
        "0"))))
View(Customer_Churn)  
# Determining baseline accuracy
Customer_Churn_freq <- Customer_Churn$Churn
cbind(frequency =
        table(Customer_Churn_freq),
      percentage = prop.table(table(Customer_Churn_freq)) * 100)
```

    ##   frequency percentage
    ## 1       495   15.71429
    ## 0      2655   84.28571

``` r
# Splitting the dataset
train_index <- createDataPartition(Customer_Churn$Churn,
                                   p = 0.80,
                                   list = FALSE)
Customer_Churn_train <- Customer_Churn[train_index, ]
Customer_Churn_test <- Customer_Churn[-train_index, ]
# Training the model; Applying 10 fold cross validation
train_control <- trainControl(method = "cv", number = 10)
Churn_model_glm <-
  train(Churn ~ ., data = Customer_Churn_train, method = "glm",
        metric = "Accuracy", trControl = train_control)
# Displaying model's performance
print(Churn_model_glm)
```

    ## Generalized Linear Model 
    ## 
    ## 2520 samples
    ##   13 predictor
    ##    2 classes: '1', '0' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 2267, 2267, 2268, 2268, 2268, 2268, ... 
    ## Resampling results:
    ## 
    ##   Accuracy   Kappa    
    ##   0.8916849  0.5131414

\#RMSE, R SQUARED AND MAE

``` r
# Loading dataset
data("BostonHousing")
#Splitting dataset
set.seed(120)
train.index <- sample(1:nrow(BostonHousing), 0.8 * nrow(BostonHousing))
Boston_Train <- BostonHousing[train_index, ]
Boston_test <- BostonHousing[-train_index, ]
# resampling using 10 fold cross validation
train_control <- trainControl(method = "cv", number = 10)
#Training the model
BostonHousing_model_lm <-
  train(medv ~ ., data = Boston_Train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)
#Display model's performance
print(BostonHousing_model_lm)
```

    ## Linear Regression 
    ## 
    ## 2520 samples
    ##   13 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 371, 372, 371, 371, 372, 372, ... 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   4.698615  0.7567805  3.425809
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

\#Area under ROC Curve

``` r
# Loading dataset
library(mlbench)
data("GermanCredit")
#Baseline Accuracy
German_Credit_freq <- GermanCredit$Class
cbind(frequency =
        table(German_Credit_freq),
      percentage = prop.table(table(German_Credit_freq)) * 100)
```

    ##      frequency percentage
    ## Bad        300         30
    ## Good       700         70

``` r
#Split Dataset
train_index <- createDataPartition(GermanCredit$Class,
                                   p = 0.8,
                                   list = FALSE)
German_Credit_train <- GermanCredit[train_index, ]
German_Credit_test <- GermanCredit[-train_index, ]
# Train model
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)
set.seed(10)

German_Credit_model <- train(Class ~ ., data = German_Credit_train, method = "knn",
                         metric = "ROC", trControl = train_control)
#Display Model's Performance
print(German_Credit_model)
```

    ## k-Nearest Neighbors 
    ## 
    ## 800 samples
    ##  61 predictor
    ##   2 classes: 'Bad', 'Good' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold) 
    ## Summary of sample sizes: 720, 720, 720, 720, 720, 720, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   k  ROC        Sens       Spec     
    ##   5  0.5562872  0.2125000  0.8339286
    ##   7  0.5619048  0.1958333  0.8803571
    ##   9  0.5647321  0.1708333  0.8821429
    ## 
    ## ROC was used to select the optimal model using the largest value.
    ## The final value used for the model was k = 9.

\#Logarithmic Loss

``` r
#Loading the dataset from mlbench
library(mlbench)
data("GermanCredit")

train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, # resampling using repeated cross validation
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)
# Train model
German_Credit_model_cart <- train(Class ~ ., data = GermanCredit, method = "rpart",
                         metric = "logLoss", trControl = train_control)
# Displaying model's performance
print(German_Credit_model_cart)
```

    ## CART 
    ## 
    ## 1000 samples
    ##   61 predictor
    ##    2 classes: 'Bad', 'Good' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 800, 800, 800, 800, 800, 800, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          logLoss  
    ##   0.01555556  0.5800414
    ##   0.03000000  0.5656368
    ##   0.04444444  0.5962752
    ## 
    ## logLoss was used to select the optimal model using the smallest value.
    ## The final value used for the model was cp = 0.03.
