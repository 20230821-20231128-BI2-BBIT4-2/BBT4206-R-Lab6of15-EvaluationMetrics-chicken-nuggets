## Loading required packages
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# Accuracy and Cohen's Kappa
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

# RMSE, R Squared and MAE
# Loading dataset
data("BostonHousing")
#Splitting dataset
set.seed(120)
train.index <- sample(1:nrow(BostonHousing), 0.8 * nrow(BostonHousing))
Boston_Train <- BostonHousing[train_index, ]
Boston_test <- BostonHousing[-train_index, ]
#Training the model
train_control <- trainControl(method = "cv", number = 10)
BostonHousing_model_lm <-
  train(medv ~ ., data = Boston_Train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)
#Display model's performance
print(BostonHousing_model_lm)

# Area under ROC Curve
# Loading dataset
library(mlbench)
data("GermanCredit")
#Baseline Accuracy
German_Credit_freq <- GermanCredit$Class
cbind(frequency =
        table(German_Credit_freq),
      percentage = prop.table(table(German_Credit_freq)) * 100)
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

#Logarithmic Loss
#Load dataset
library(mlbench)
data("GermanCredit")
# Train model
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)
German_Credit_model_cart <- train(Class ~ ., data = GermanCredit, method = "rpart",
                         metric = "logLoss", trControl = train_control)
# Display model's performance
print(German_Credit_model_cart)










