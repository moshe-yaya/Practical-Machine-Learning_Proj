setwd("D:/Documents/coursera/Practical Machine Learning/Project")

library(ggplot2)
library(caret)

FullTrain <- read.csv("pml-training.csv")
FullTest <- read.csv("pml-testing.csv")

#set.seed(100)
#samplesize <- rep(NA,2000)
#samplesize <- sample(1:19622,2000)
#samplesize <- sort(samplesize)
#train <- FullTrain[samplesize,]

train <- FullTrain
NameList <- NA

NameList <- c('pitch_belt','yaw_belt','total_accel_belt','roll_arm','pitch_arm  yaw_arm','total_accel_arm'
              ,'total_accel_dumbbell','roll_forearm','pitch_forearm','yaw_forearm','total_accel_forearm','accel_arm_x'
              ,'accel_dumbbell_z','gyros_arm_x','gyros_dumbbell_z','magnet_dumbbell_x'
              ,'accel_forearm_y','magnet_forearm_y','accel_belt_z','magnet_belt_x','roll_belt','gyros_forearm_y','classe')
train<- train[,colnames(train) %in% NameList] 

M <- abs(cor(train[,-22]))
diag(M) <- 0
which(M > 0.8, arr.ind=T) # total_accel_belt,pitch_belt,gyros_dumbbell_z

NameList <- NA
NameList <- c('pitch_belt','yaw_belt','total_accel_belt','roll_arm','pitch_arm  yaw_arm','total_accel_arm'
              ,'total_accel_dumbbell','roll_forearm','pitch_forearm','yaw_forearm','total_accel_forearm','accel_arm_x'
              ,'accel_dumbbell_z','gyros_arm_x','gyros_dumbbell_z','magnet_dumbbell_x'
              ,'accel_forearm_y','magnet_forearm_y','classe')
train<- train[,colnames(train) %in% NameList] 
FullTest  <- FullTest [,colnames(FullTest) %in% NameList] 

train$classe <- as.factor(train$classe)  



set.seed(19031)
inTrain <- createDataPartition(train$classe, p = 0.75, list=FALSE)
training <- train[inTrain, ]
testing  <- train[-inTrain, ]


modFit <- train(classe ~., method="rf", data=training, trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )


pred <- predict(modFit, testing)
confusionMatrix(pred, testing$classe)



testingPred <- predict(modFit,FullTest )
testingPred

library(rattle)
library(rpart.plot)
fancyRpartPlot(modFit$finalModel)




#---------------
answers = as.character(testingPred)
class(answers)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)






