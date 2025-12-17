#Load Libraries
install.packages("e1071")
install.packages("caret")
install.packages("dplyr")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("xgboost")
install.packages("randomForest")

library(e1071)
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)

#Load data
data<-read.csv("used_car_dataset.csv",na.strings = "")
#Remove NAs
data$kmDriven[is.na(data$kmDriven)]<-mean(data$kmDriven)

#kmDriven column is chr and has values that needs to be trimmed and turn it into integer
data$kmDriven <- gsub(" km", "", data$kmDriven)
data$kmDriven <- gsub(",", "", data$kmDriven)
data$kmDriven <- as.integer(data$kmDriven)

#Checking that age column matches the year column
data$Age[data$Age!=2025-data$Year]<-2025-data$Year

#Trimming AskPrice column from unhelpful characters and turning it into integer type
data$AskPrice <- gsub("₹", "", data$AskPrice)
data$AskPrice <- gsub(",", "", data$AskPrice)
data$AskPrice <- as.integer(data$AskPrice)

# Convert AskPrice to Classification Target
data$price_class <- cut(
  data$AskPrice,
  breaks = quantile(data$AskPrice, probs = c(0, 0.33, 0.66, 1)),
  labels = c("Low", "Medium", "High"),
  include.lowest = TRUE
)
data$price_class <- as.factor(data$price_class)

# Remove numeric price column (Naive Bayes = classification)
data$AskPrice <- NULL

# Convert target to factor
data$price_class <- as.factor(data$price_class)

#Removing unhelpful column from the dataset for better normalization
data$AdditionInfo<-NULL

# Reduce high-cardinality categorical variables
# Keep top 15 brands, rest as Other
top_brands <- names(sort(table(data$Brand), decreasing = TRUE))[1:15]
data$Brand <- ifelse(data$Brand %in% top_brands, as.character(data$Brand), "Other")
data$Brand <- as.factor(data$Brand)

# Keep top 25 models, rest as Other
top_models <- names(sort(table(data$model), decreasing = TRUE))[1:25]
data$model <- ifelse(data$model %in% top_models, as.character(data$model), "Other")
data$model <- as.factor(data$model)

#Normalizing the value of the character columns so it helps in the training and testing
data$Transmission <- as.factor(data$Transmission)
data$Owner <- as.factor(data$Owner)
data$FuelType <- as.factor(data$FuelType)

#Turning the date from a character column into a date type column
data$PostedDate <- as.Date(paste0("01-", data$PostedDate),format = "%d-%b-%y")

#Remove Nulls
data <- na.omit(data)

#Training Data
train_data  <- rbind(data[1:1500,], data[3001:4500,])
train_data  <- rbind(train_data , data[6001:7500,])
train_data  <- rbind(train_data , data[8501:9535,])

#Testing Data
test_data  <- rbind(data[1501:3000,], data[4501:6000,])
test_data  <- rbind(test_data , data[7501:8500,])

str(data)
head(data)



# Naive Bayes

#1: Naive Bayes Model (FIXED FORMULA)
classyPrice <- naiveBayes(price_class ~ Age + kmDriven + Brand + model + FuelType + Transmission + Owner, data = train_data)

#2: Prediction
PricePrediction <- predict(classyPrice, newdata = test_data)

#3: Naive Bayes Evaluation
table(PricePrediction)
table(PricePrediction, test_data$price_class)
conf_matrix <- table(PricePrediction, test_data$price_class)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Model Accuracy =", round(accuracy * 100, 2), "%"))



# Decision Tree

tree_model <- rpart(price_class ~ Age + kmDriven + Brand + model + FuelType + Transmission + Owner,
                    data = train_data,
                    method = "class",
                    control = rpart.control(cp = 0.005, maxdepth = 15))

#1: Plot Tree
rpart.plot(tree_model, main = "Used Cars Price Decision Tree")

#2: Decision Tree Prediction
tree_pred <- predict(tree_model, test_data, type = "class")

#3: Decision Tree Evaluation
tree_conf <- confusionMatrix(tree_pred, test_data$price_class)
print("Decision Tree Confusion Matrix:")
print(tree_conf)




# Random Forest Model

set.seed(123)

rf_model <- randomForest(
  price_class ~ Age + kmDriven + Brand + model + FuelType + Transmission + Owner,
  data = train_data,
  ntree = 200,
  importance = TRUE
)

rf_pred <- predict(rf_model, newdata = test_data)

rf_conf_matrix <- table(rf_pred, test_data$price_class)
rf_accuracy <- sum(diag(rf_conf_matrix)) / sum(rf_conf_matrix)

print(rf_conf_matrix)                 
print(paste("Random Forest Accuracy =", round(rf_accuracy * 100, 2), "%"))  

varImpPlot(rf_model)            



# XGBoost Model

#1: Prepare feature matrices
x_train <- train_data[, c("Age", "kmDriven", "Brand", "model", "FuelType", "Transmission", "Owner")]
x_test  <- test_data[,  c("Age", "kmDriven", "Brand", "model", "FuelType", "Transmission", "Owner")]

#2: One-hot encoding for categorical variables
dummies <- dummyVars(~ ., data = x_train)

train_matrix <- predict(dummies, newdata = x_train)
test_matrix  <- predict(dummies, newdata = x_test)

#3: Convert labels to 0,1,2 (required by XGBoost)
y_train <- as.numeric(train_data$price_class) - 1
y_test  <- as.numeric(test_data$price_class) - 1


#4: Convert to DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
dtest  <- xgb.DMatrix(data = test_matrix, label = y_test)


#5: XGBoost parameters
params <- list(
  objective = "multi:softmax",
  num_class = 3,
  eval_metric = "merror",
  max_depth = 6,
  learning_rate = 0.3
)


#6: Train model
set.seed(123)

xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100
)


#7: Prediction & Evaluation
xgb_pred <- predict(xgb_model, dtest)

xgb_conf_matrix <- table(
  factor(xgb_pred, levels = c(0, 1, 2),
         labels = levels(test_data$price_class)),
  test_data$price_class
)

xgb_accuracy <- sum(diag(xgb_conf_matrix)) / sum(xgb_conf_matrix)

print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)

print(paste("XGBoost Accuracy =", round(xgb_accuracy * 100, 2), "%"))