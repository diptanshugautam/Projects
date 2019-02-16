#Setting Working Directory
setwd("E:/Notes and ppts/Edwisor/Projects/Project 3")

#Loading train and test data
train = read.csv("Train_data.csv")
test = read.csv("Test_data.csv")


#converting the data type
train$area.code = as.factor(train$area.code)
test$area.code = as.factor(test$area.code)

#Combining Train and test data
data = rbind(train, test)

#Exploring the data
str(data)

#Missing Value Analysis
sum(is.na(data))


#splitting data into factor and numeric type
factor.index = sapply(data, is.factor)
numeric.index = sapply(data, is.numeric)
factor.data = data[,factor.index]
numeric.data = data[,numeric.index]


#Outlier Analysis
cnames = colnames(numeric.data)
for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(data))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="Churn")+
           ggtitle(paste("Box plot of Churn for",cnames[i])))
}

## Plotting plots together
gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn7,gn8,gn9,ncol=3)
gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)

#replacing all the outliers with NA and imputing them using knn method


columns = colnames(data)
for(i in columns){
    val = data[,i][data[,i] %in% boxplot.stats(data[,i])$out]
    print(length(val))
    data[,i][data[,i] %in% val] = NA
  }
sum((is.na(data)))

#number of outliers detected and removed = 1080
library(DMwR)
data = knnImputation(data, k = 5)

#Detecting correalation and dependence between variables 

#plotting correlogram
corrgram(numeric.data, order = FALSE, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")
plot(data$total.day.minutes, data$total.day.charge)

corre_ana = cor(numeric.data, numeric.data)
write.csv(corre_ana, "Correlation Analysis.csv")

plot(data2$total.eve.minutes,data2$total.eve.charge, xlab = "Total Evening Minutes",ylab = "Total Evening Charge")

#Chi Square Test of Independence

for (i in 1:21)
{
  print(names(data)[i])
  print(chisq.test(table(factor.data$Churn,data[,i])))
}


#Dimensionality Reduction
data = subset(data, select = -c(area.code,phone.number,total.day.minutes,total.eve.minutes,total.night.minutes))

#recreating original train and test data partition

train = data[1:3333,]
test = data[3334:5000,]

#Creating Model

#Decision Tree Model
library(C50)
library(rpart)
classifier = C5.0(Churn~., train, trials = 100, rules = TRUE)
summary(classifier)
write(capture.output(summary(classifier)), "C50 Output.txt")
regressor = rpart(Churn~., data = train)
C50_predict = predict(classifier, test[,-16], type = "class")
table(C50_predict)
table(test$Churn)

#Plotting Decision Tree and result
regressor = rpart(Churn~., data = train)
rpart.plot::rpart.plot(regressor, box.palette = "RdBu", shadow.col = "gray", nn=TRUE)
plot(C50_predict, test[,16], xlab = "Predicted Churn Score", ylab = "Actual Churn Score")


##Evaluate the performance of classification model
ConfMatrix_C50 = table(test$Churn, C50_predict)
library(caret)
confusionMatrix(ConfMatrix_C50)

#Random Forest Model
library(randomForest)
RF_model = randomForest(Churn~.,data = train, ntree = 100)
pred = predict(RF_model, test[,-16], type = "class")
table(pred)

#Extract rules fromn random forest
#transform rf object to an inTrees' format
library(inTrees)
treeList = RF2List(RF_model) 
 
#Extract rules
exec = extractRules(treeList, train[,-16])
 
#Make rules more readable:
readableRules = presentRules(exec, colnames(train))

#Get rule metrics
ruleMetric = getRuleMetric(exec, train[,-16], train$responded)  # get rule metrics

#Writing rules to a text file
write(capture.output(readableRules), "Random Forest Rules.txt")

#Get rule metrics
ruleMetric = getRuleMetric(exec, train[,-16], train$Churn)
write(capture.output(ruleMetric), "Random Forest Rule metric.txt")

#Plotting Random Forest and result
plot(RF_model)
plot(pred, test[,16], xlab = "Predicted Churn Score", ylab = "Actual Churn Score")

##Evaluate the performance of classification model
ConfMatrix_RF = table(test$Churn, pred)
confusionMatrix(ConfMatrix_RF)

#Naive Bayes
library(e1071)
class_nb = naiveBayes(Churn~., data = train)
predict_nb = predict(class_nb, test[,-16], type = 'class')

#Plotting Naive Bayes result
plot(predict_nb, test[,16], xlab = "Predicted Churn Score", ylab = "Actual Churn Score")

#Evaluating Naive Bayes by confusion matrix
ConfMatrix_nb = table(test$Churn, predict_nb)
confusionMatrix(ConfMatrix_nb)
