#Load data
data<-read.csv("used_car_dataset.csv",na.strings = "")
#Remove NAs
data$kmDriven[is.na(data$kmDriven)]<-mode(data$kmDriven)

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

# Remove numeric price column (Naive Bayes = classification)
data$AskPrice <- NULL

# Convert target to factor
data$price_class <- as.factor(data$price_class)

#Removing unhelpful column from the dataset for better normalization
data$AdditionInfo<-NULL

#Normalizing the value of the character columns so it helps in the training and testing
data$Brand <- as.factor(data$Brand)
data$model <- as.factor(data$model)
data$Transmission <- as.factor(data$Transmission)
data$Owner <- as.factor(data$Owner)
data$FuelType <- as.factor(data$FuelType)

#Turning the date from a character column into a date type column
data$PostedDate <- as.Date(paste0("01-", data$PostedDate),format = "%d-%b-%y")

str(data)
head(data)



# Naive Bayes
install.packages("e1071")
library(e1071)

#1: Training Data (rbind – lab style)
newdt <- rbind(data[1:1500,], data[3001:4500,])
newdt <- rbind(newdt, data[6001:7500,])
newdt <- rbind(newdt, data[8501:9582,])

#2: Testing Data (rbind – lab style)
testdt <- rbind(data[1501:3000,], data[4501:6000,])
testdt <- rbind(testdt, data[7501:8500,])

#3: Naive Bayes Model (FIXED FORMULA)
classyPrice <- naiveBayes(price_class ~ Age + kmDriven + FuelType + Transmission + Owner, data = newdt)

#4: Prediction
PricePrediction <- predict(classyPrice, newdata = testdt)

#5: Result Tables
table(PricePrediction)
table(PricePrediction, testdt$price_class)

#6: Accuracy
conf_matrix <- table(PricePrediction, testdt$price_class)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)

print(paste("Model Accuracy =", round(accuracy * 100, 2), "%"))




