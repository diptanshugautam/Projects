#Setting the working directory
setwd("E:\\Notes and ppts\\Edwisor\\Projects\\Project 2")
getwd()

#Uploading the data
day = read.csv("day.csv")

#removing date, temperature, windspeed and humidity column
day = day[,-c(2,10,12,13)]

#exploratory data analysis
str(day)
c = 2:8

#converting the data type
day[c] = lapply(day[c], as.factor)


#segregating numeric and factor data type
numeric.index = sapply(day, is.numeric)
numeric_data = day[,numeric.index]
factor.index = sapply(day, is.factor)
factor_data = day[,factor.index]

#Missing Value Analysis
sum(is.na(day))

#Outlier Analysis
cnames = colnames(numeric_data)
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "cnt"), data = subset(day))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="cnt")+
           ggtitle(paste("Box plot of Count for",cnames[i])))
}

## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,ncol=2)
gridExtra::grid.arrange(gn3,gn4,gn5,ncol=3)

#replacing all the outliers with NA and imputing them using knn method
val = day$casual[day$casual %in% boxplot.stats(day$casual)$out]
day$casual[day$casual %in% val] = NA

sum(is.na(day$casual))
#Number of outlier detected and removed = 44

#Imputing NA values with KNN imputation
library(DMwR)
day = knnImputation(day, k = 5)

#Creating Multiple Histograms
library('psych')
multi.hist(numeric_data, main = NA, dcol = c("Blue", "Red"), dlty = c("solid","solid"), bcol = "grey95")

#Detecting Corelation between variables
library(corrgram)
corrgram(numeric_data, order = FALSE, upper.panel = panel.pie, text.panel = panel.text,
         main = "Correlation Plot")

corre_ana = cor(numeric_data)
write.csv(corre_ana, "Correlation Analysis.csv")

#Normality Check
hist(day$casual)
hist(day$registered)

#Testing the dependence of variable using multiple linear regresssion

mult_lin = lm(cnt~., numeric_data[,-1])
summary(mult_lin)

#Testing the presence of multi collinearity
car::vif(mult_lin)

#ANOVA Testing
anova = aov(day$cnt~.,factor_data)
summary(anova)


#Creating Train and Test Data Set
library(caret)
train.index = createDataPartition(day$cnt, p=0.8, list = FALSE)
train = day[train.index,]
test = day[-train.index,]

#Decision Tree Model
library(rpart)
set.seed(1234)
regressor = rpart(cnt~., data = train)
rpart.plot::rpart.plot(regressor, box.palette = "RdBu", shadow.col = "gray", nn=TRUE)

#Predicting the test data
y_pred = predict(regressor, test)

##Evaluate the performance of regression model
devi = abs((y_pred - test$cnt)/test$cnt)
concat = as.data.frame(cbind(test$cnt, y_pred, devi))
colnames(concat) = c("Actual", "Predicted", "Deviation")

MAPE = (sum(devi)/length(y_pred))*100
#Mean Absolute Percentage error = 9.82%



#Random Forest Model
library(randomForest)
regres = randomForest(cnt~., data = train, ntree = 50)

#Plotting Random Forest and Result
plot(regres)

#predicting the test data set
pred = predict(regres, test)


#Evaluating the Random Forest Result
devi_RF = abs((pred - test$cnt)/test$cnt)
concat_RF = as.data.frame(cbind(test$cnt, pred, devi_RF))
colnames(concat_RF) = c("Actual", "Predicted", "Deviation")

MAPE_RF = (sum(devi_RF)/length(y_pred))*100
MAPE_RF
#Mean Absolute Percentage Error: 6.05%


#Support Vector Regression
library(e1071)

#Develop model
svm_model = svm(formula = cnt ~ ., data = train, type = "eps-regression")
plot(svm_model)

#predict on test cases
svm_predict = predict(svm_model, test[,1:12])

#Evaluating the Support Vector Result
devi_svm = abs((svm_predict - test$cnt)/test$cnt)
concat_svm = as.data.frame(cbind(test$cnt, svm_predict, devi_RF))
colnames(concat_RF) = c("Actual", "Predicted", "Deviation")

MAPE_svm = (sum(devi_svm)/length(y_pred))*100
MAPE_svm
#Mean Absolute Percentage Error: 3.767%








