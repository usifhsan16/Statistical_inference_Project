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