# Practical-Machine-Learning_Proj
Peer Assessments /Prediction Assignment Writeup

Practical Machine Learning / Peer Assessments
========================================================
#Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement ??– a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this data set, the participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants toto predict the manner in which praticipants did the exercise.

The dependent variable or response is the “classe” variable in the training set


# loading data

```r
FullTrain <- read.csv("pml-training.csv")
FullTest <- read.csv("pml-testing.csv")
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.1.2
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.4.1 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```
* set the 'wd' where the files Saved

## data cleaning explanation
* Each column that contains the type of time, has been removed. Because it is necessary to predict one observation, is not necessary in all fields
Which contain information on operating time, such as "time window"
* Every parameter which contains 3 coordinates X, Y, Z . analysis was conducted with the highest variance coordinates,With the idea that most human movement when training is done in one axis.
* High correlation variables are reduced by various to one varible




```r
# Choosing 1 coordinates from x,y,z
var(FullTrain$gyros_arm_x)
```

```
## [1] 3.974428
```

```r
var(FullTrain$gyros_arm_y)
```

```
## [1] 0.7248659
```

```r
var(FullTrain$gyros_arm_z)
```

```
## [1] 0.3059957
```

```r
# * gyros_arm_x wos Selected

train <- FullTrain
NameList <- NA
#list of All relevant parameters
NameList <- c('pitch_belt','yaw_belt','total_accel_belt','roll_arm','pitch_arm  yaw_arm','total_accel_arm'
              ,'total_accel_dumbbell','roll_forearm','pitch_forearm','yaw_forearm','total_accel_forearm','accel_arm_x'
              ,'accel_dumbbell_z','gyros_arm_x','gyros_dumbbell_z','magnet_dumbbell_x'
              ,'accel_forearm_y','magnet_forearm_y','accel_belt_z','magnet_belt_x','roll_belt','gyros_forearm_y','classe')
train<- train[,colnames(train) %in% NameList] 

# reduced High correlation variables
M <- abs(cor(train[,-22]))
diag(M) <- 0
which(M > 0.8, arr.ind=T) # total_accel_belt,pitch_belt,gyros_dumbbell_z
```

```
##                  row col
## yaw_belt           3   1
## total_accel_belt   4   1
## accel_belt_z       5   1
## magnet_belt_x      6   2
## roll_belt          1   3
## roll_belt          1   4
## accel_belt_z       5   4
## roll_belt          1   5
## total_accel_belt   4   5
## pitch_belt         2   6
```

```r
# remove 'accel_belt_z','magnet_belt_x','roll_belt','gyros_forearm_y'

#-------------
NameList <- NA
NameList <- c('pitch_belt','yaw_belt','total_accel_belt','roll_arm','pitch_arm  yaw_arm','total_accel_arm'
              ,'total_accel_dumbbell','roll_forearm','pitch_forearm','yaw_forearm','total_accel_forearm','accel_arm_x'
              ,'accel_dumbbell_z','gyros_arm_x','gyros_dumbbell_z','magnet_dumbbell_x'
              ,'accel_forearm_y','magnet_forearm_y','classe')
train<- train[,colnames(train) %in% NameList] 
FullTest  <- FullTest [,colnames(FullTest) %in% NameList] 

train$classe <- as.factor(train$classe)  
# 17 explanatory parameterswere selected  < NameList >
```


# Create cross validation set

The training set is divided in two parts, one for training and the other for cross validation


```r
set.seed(190)
inTrain <- createDataPartition(train$classe, p = 0.75, list=FALSE)
training <- train[inTrain, ]
testing  <- train[-inTrain, ]
```

# Train model

Train model with random forest due to its highly accuracy rate. The model is build on a training set of 17 variables from the initial 160. Cross validation is used as train control method

```r
set.seed(16331)
modFit <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```


# Accuracy on  cross validation set

```r
pred <- predict(modFit, testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    8    0    0    0
##          B    2  935    2    0    5
##          C    0    3  852    9    3
##          D    0    3    1  795    4
##          E    0    0    0    0  889
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9918          
##                  95% CI : (0.9889, 0.9942)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9897          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9986   0.9852   0.9965   0.9888   0.9867
## Specificity            0.9977   0.9977   0.9963   0.9980   1.0000
## Pos Pred Value         0.9943   0.9905   0.9827   0.9900   1.0000
## Neg Pred Value         0.9994   0.9965   0.9993   0.9978   0.9970
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2841   0.1907   0.1737   0.1621   0.1813
## Detection Prevalence   0.2857   0.1925   0.1768   0.1637   0.1813
## Balanced Accuracy      0.9981   0.9915   0.9964   0.9934   0.9933
```
# RESULTS
Predictions on the real testing set
* on the  Cross validation thr model predect Accuracy : 0.9927   


```r
testingPred <- predict(modFit,FullTest )
testingPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

