# Clearing the R environment
rm(list=ls(all=T))
#loading the required libraries
library(dplyr)
library(DMwR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(glmnet)
library(ROCR)
library(randomForest)
library(C50)
library(e1071)
library(ROSE) #Randomly oversampling examples
drat:::addRepo("dmlc")
library(xgboost)
library(h2o)
library(corrplot)
library(ggplot2)
require("scatterplot3d")
require("rgl")
require("RColorBrewer")
library(rpart)
library(ada) 
library(kernlab)
library(doSNOW)

#Reading and analysing the data
ptrain<- read.csv("Train.csv",header = T,sep = ",")
adata<- read.csv("Train_AdditionalData.csv",header=T,sep = ",")
str(ptrain)
str(adata)
#creating new dataset by joining both

####This is all for train dataset
final_train<-ptrain
sum(is.na(ptrain$ID))
sum(is.na(adata$TestA))
nrow(ptrain)
a=adata$TestA
b=adata$TestB
a=data.frame(a)
b=data.frame(b)
a=na.omit(a)
nrow(a)
a$ID=a$a
mergeddata <- merge(ptrain,a,by=c("ID"),all=TRUE)
head(mergeddata)
b$ID=b$b
mergeddata <- merge(mergeddata,b,by=c("ID"),all=TRUE)
head(mergeddata)
mergeddata$a[is.na(mergeddata$a)] <- 0
head(mergeddata)
mergeddata$b[is.na(mergeddata$b)] <- 0
head(mergeddata)
mergeddata$a[mergeddata$a>0]= 1
head(mergeddata)
mergeddata$b[mergeddata$b>0]= 1
head(mergeddata)
View(mergeddata)

####This is all for test dataset
ptest<- read.csv("Test.csv",header = T,sep = ",")
atest<- read.csv("Test_AdditionalData.csv",header=T,sep = ",")
final_test<-ptest
sum(is.na(ptest$ID))
sum(is.na(atest$TestA))
nrow(ptest)
a=atest$TestA
b=atest$TestB
a=data.frame(a)
b=data.frame(b)
b=na.omit(b)
nrow(b)
a$ID=a$a
Test_mergeddata <- merge(ptest,a,by=c("ID"),all=TRUE)
head(Test_mergeddata)
b$ID=b$b
Test_mergeddata <- merge(Test_mergeddata,b,by=c("ID"),all=TRUE)
head(Test_mergeddata)
Test_mergeddata$a[is.na(Test_mergeddata$a)] <- 0
head(Test_mergeddata)
Test_mergeddata$b[is.na(Test_mergeddata$b)] <- 0
head(Test_mergeddata)
Test_mergeddata$a[Test_mergeddata$a>0]= 1
head(Test_mergeddata)
Test_mergeddata$b[Test_mergeddata$b>0]= 1
head(Test_mergeddata)
View(Test_mergeddata)

#####Done with Data Gathering
##Now Data PreProcessing
##check for class imbalance
barplot(table(mergeddata$y))
#no class imbalance
##checking if any target variable is missing in train data
sum(is.na(mergeddata$y))   #0
sum(is.na(mergeddata$Number.of.Cylinders))
nrow(mergeddata)
158/3156    #5%
sum(is.na(mergeddata$material.grade)) #5%
sum(is.na(mergeddata$Lubrication)) #5%
sum(is.na(mergeddata$Valve.Type)) #5%
sum(is.na(mergeddata$Bearing.Vendor))  #5%
sum(is.na(mergeddata$Fuel.Type)) #5%
sum(is.na(mergeddata$Compression.ratio)) #5%
sum(is.na(mergeddata$cam.arrangement)) #5%
sum(is.na(mergeddata$Cylinder.arragement)) #5%
sum(is.na(mergeddata$Turbocharger))
sum(is.na(mergeddata$Varaible.Valve.Timing..VVT.))
sum(is.na(mergeddata$Cylinder.deactivation))
sum(is.na(mergeddata$Direct.injection))
sum(is.na(mergeddata$main.bearing.type))
sum(is.na(mergeddata$displacement))
sum(is.na(mergeddata$piston.type))
sum(is.na(mergeddata$Max..Torque))
sum(is.na(mergeddata$Peak.Power))
sum(is.na(mergeddata$Crankshaft.Design))
sum(is.na(mergeddata$Liner.Design.))
sum(is.na(mergeddata$a))
sum(is.na(mergeddata$b))

###  So in every attribute values are missing #####
#central imputation
mergeddata2=centralImputation(mergeddata)
sum(is.na(mergeddata2))

#TypeCasting
str(mergeddata2)
mergeddata2$a=as.factor(mergeddata2$a)
mergeddata2$b=as.factor(mergeddata2$b)
Test_mergeddata2=Test_mergeddata
Test_mergeddata2$a=as.factor(Test_mergeddata2$a)
Test_mergeddata2$b=as.factor(Test_mergeddata2$b)
Test_mergeddata2=centralImputation(Test_mergeddata2)
write.csv(mergeddata2, file = "mergeddata.csv",col.names = TRUE,row.names = FALSE)
write.csv(Test_mergeddata2, file = "Test_mergeddata.csv",col.names = TRUE,row.names = FALSE)
###Model Building
set.seed(1234)
index_train <- createDataPartition(mergeddata2$y, p = 0.7, list = F)
pre_train1 <- mergeddata2[index_train, ]
pre_test1 <- mergeddata2[-index_train, ]
dim(pre_train1)
dim(pre_test1)
#Logistic regression
log_reg1 <- glm(y ~ ., data = pre_train1, family = "binomial")
summary(log_reg1)
proab_train1 <- predict(log_reg1, type = "response")
pred1 <- prediction(proab_train1, pre_train1$y)
perf1 <- performance(pred1, measure="tpr", x.measure="fpr")
plot(perf1, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc1 <- performance(pred1, measure="auc")
auc1 <- perf_auc1@y.values[[1]]
print(auc1)
cutoffs1 <- data.frame(cut= perf1@alpha.values[[1]], fpr= perf1@x.values[[1]], 
                       tpr=perf1@y.values[[1]])
cutoffs1 <- cutoffs1[order(cutoffs1$tpr, decreasing=TRUE),]
View(cutoffs1)
plot(perf1, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
pred_train1 <- ifelse(proab_train1> 0.65, "pass", "fail")
table(pre_train1$y,pred_train1)
confusionMatrix(pred_train1, pre_train1$y, positive = "pass")
#on validation
prob_test1 <- predict(log_reg1, pre_test1, type = "response")
preds_test1 <- ifelse(prob_test1 > 0.65, "pass", "fail")
confusionMatrix(preds_test1, pre_test1$y, positive = "pass")

#########Prediction on original Test data with Model 1##########
prob_org1 <- predict(log_reg1, Test_mergeddata2, type = "response")
preds_org1 <- ifelse(prob_org1 > 0.65, "pass", "fail")
preds_org1=data.frame(preds_org1)
write.csv(preds_org1, file = "Data1.csv",col.names = TRUE,row.names = FALSE)


#######Naive baiese
nb_model1 <- naiveBayes(y~.,data = pre_train1)
nb_model1
nb_test_predict1 <- predict(nb_model1,pre_test1)
confusionMatrix(nb_test_predict1, pre_test1$y, positive = "pass")

##step AIC
#stepAICofTrain1=stepAIC(log_reg1,direction = "backward")
log_reg1_1 <- glm(y~Lubrication + Fuel.Type + cam.arrangement + Cylinder.deactivation +displacement + Max..Torque + Peak.Power + Liner.Design. + a + b, data = pre_train1, family = "binomial")
summary(log_reg1)
proab_train1.1 <- predict(log_reg1_1, type = "response")
pred1.1 <- prediction(proab_train1.1, pre_train1$y)
pred1.1 <- prediction(proab_train1.1, pre_train1$y)
perf1.1 <- performance(pred1.1, measure="tpr", x.measure="fpr")
plot(perf1.1, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc1.1 <- performance(pred1.1, measure="auc")
auc1.1 <- perf_auc1.1@y.values[[1]]
print(auc1.1)
#so not better model than that


###Decision Tress
#C5.0
pre_train1_ID=pre_train1$ID
pre_train1=pre_train1[ , -which(names(pre_train1) %in% c("ID"))]
pre_test1=pre_test1[ , -which(names(pre_test1) %in% c("ID"))]
Test_mergeddata2=Test_mergeddata2[ , -which(names(Test_mergeddata2) %in% c("ID"))]
DT_C50_1 <- C5.0(y~.,data=pre_train1)
summary(DT_C50_1)
pred_Train1_3 = predict(DT_C50_1,newdata=pre_train1, type="class")
pred_Test1_3 = predict(DT_C50_1, newdata=pre_test1, type="class")
confusionMatrix(pre_train1$y,pred_Train1_3,positive = "pass")
confusionMatrix(pre_test1$y,pred_Test1_3,positive = "pass")
C5imp(DT_C50_1, pct=TRUE)

DT_C50_2 = C5.0(y~ Fuel.Type+cam.arrangement+Peak.Power+b+a,data=pre_train1)
summary(DT_C50_2)
pred_Train1_4 = predict(DT_C50_2,newdata=pre_train1, type="class")
pred_Test1_4 = predict(DT_C50_2, newdata=pre_test1, type="class")
confusionMatrix(pre_train1$y,pred_Train1_4,positive = "pass")
confusionMatrix(pre_test1$y,pred_Test1_4,positive = "pass")

####C.50 predictions on Original test Data
pred_org_Test = predict(DT_C50_1, newdata=Test_mergeddata2, type="class")
pred_org_Test=data.frame(pred_org_Test)
write.csv(preds_org1, file = "Data2.csv",col.names = TRUE,row.names = FALSE)


#CART
DT_rpart_1<-rpart(y~.,data=pre_train1,method="class")
DT_rpart_1$cptable
ptree<- prune(DT_rpart_1,cp= DT_rpart_1$cptable[which.min(DT_rpart_1$cptable[,"xerror"]),"CP"])
pred_Train1_5 = predict(DT_rpart_1,newdata=pre_train1, type="class")
pred_Test1_5 = predict(DT_rpart_1, newdata=pre_test1, type="class")
confusionMatrix(pre_train1$y,pred_Train1_5,positive = "pass")
confusionMatrix(pre_test1$y,pred_Test1_5,positive = "pass")
pred_Train1_6 = predict(ptree,newdata=pre_train1, type="class")
pred_Test1_6 = predict(ptree, newdata=pre_test1, type="class")
confusionMatrix(pre_train1$y,pred_Train1_6,positive = "pass")
confusionMatrix(pre_test1$y,pred_Test1_6,positive = "pass")

#Random Forest
model_rf1 <- randomForest(y~ . , pre_train1,ntree = 80,mtry = 5)
summary(model_rf1)
importance(model_rf1)
varImpPlot(model_rf1)
preds_trf1=predict(model_rf1,pre_train1)
preds_rf1 <- predict(model_rf1, pre_test1)
confusionMatrix(preds_rf1, pre_test1$y,positive = "pass")
###predictions on original test data###
Test_mergeddata4=Test_mergeddata2
Test_mergeddata4$y="Pass"
Test_mergeddata4$y[1]="fail"
Test_mergeddata4$y=as.factor(Test_mergeddata4$y)
Test_mergeddatay=Test_mergeddata4$y
Test_mergeddata4=Test_mergeddata4[ , -which(names(Test_mergeddata4) %in% c("y"))]
Test_mergeddata4=cbind.data.frame(Test_mergeddatay,Test_mergeddata4)
Test_mergeddata4=rename(Test_mergeddata4, y = Test_mergeddatay)
preds_org1_rf1=predict(model_rf1,Test_mergeddata4)


model_rf1_1 <-randomForest(y~. ,pre_train1,ntree=80,mtry=4)
preds_rf1_1 <- predict(model_rf1_1, pre_test1)
confusionMatrix(preds_rf1_1, pre_test1$y,positive = "pass")

#####Boosting###########
#adaboost
###Boosting
#ada
ada1 = ada(y ~ ., iter = 20,data = pre_train1, loss="logistic") 
ada1
pred7 = predict(ada1, pre_test1);
pred7 
confusionMatrix(pre_test1$y,pred7,positive = "pass")

ada2=ada(y ~ ., iter = 10,data = pre_train1, loss="logistic") 
ada2
pred8 = predict(ada2, pre_test1);
pred8 
confusionMatrix(pre_test1$y,pred8,positive = "pass")

#GBM
#GBM
h2o.init(nthreads = -1, max_mem_size = "6g")

train1.hex <- as.h2o(x = pre_train1, destination_frame = "train1.hex")
ntrees_opt <- c(5, 10, 15, 20, 30)
maxdepth_opt <- c(2, 3, 4, 5)
learnrate_opt <- c(0.01, 0.05, 0.1, 0.15 ,0.2, 0.25)
hyper_parameters <- list(ntrees = ntrees_opt, 
                         max_depth = maxdepth_opt, 
                         learn_rate = learnrate_opt)
grid_GBM1 <- h2o.grid(algorithm = "gbm", grid_id = "grid_GBM.hex",
                      hyper_params = hyper_parameters, 
                      y = "y", x = setdiff(names(train1.hex), "y"),
                      training_frame = train1.hex)
rm(ntrees_opt, maxdepth_opt, learnrate_opt, hyper_parameters)
summary(grid_GBM1)
grid_GBM_models1 <- lapply(grid_GBM1@model_ids, 
                           function(model_id) { h2o.getModel(model_id) })

find_Best_Model <- function(grid_models1){
  best_model = grid_models1[[1]]
  best_model_AUC = h2o.auc(best_model)
  for (i in 2:length(grid_models1)) 
  {
    temp_model = grid_models1[[i]]
    temp_model_AUC = h2o.auc(temp_model)
    if(best_model_AUC < temp_model_AUC)
    {
      best_model = temp_model
      best_model_AUC = temp_model_AUC
    }
  }
  return(best_model)
}

best_GBM_model = find_Best_Model(grid_GBM_models1)

rm(grid_GBM_models1)
best_GBM_model_AUC = h2o.auc(best_GBM_model)
best_GBM_model

best_GBM_model@parameters
varImp_GBM <- h2o.varimp(best_GBM_model)
varImp_GBM
test1.hex <- as.h2o(x = pre_test1, destination_frame = "test1.hex")
predict1.hex = h2o.predict(best_GBM_model, 
                           newdata = test1.hex[,setdiff(names(test1.hex), "y")])

data_GBM1 = h2o.cbind(test1.hex[,"y"], predict1.hex)
pred_GBM1 = as.data.frame(data_GBM1)
h2o.shutdown(T)
confusionMatrix(pred_GBM1$y,pred_GBM1$predict,positive = "pass")


##SVM
dummies=dummyVars(y~.,data=pre_train1)
str(dummies)
summary(dummies)
x.train=predict(dummies, newdata = pre_train1)
y.train=pre_train1$y
x.test = predict(dummies, newdata = pre_test1)
y.test = pre_test1$y
#1
svm_model1  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "linear", cost = 10)
summary(svm_model1)
predict_train=predict(svm_model1,newdata = x.train)
predict_test=predict(svm_model1,newdata = x.test)
confusionMatrix(pre_train1$y,predict_train,positive = "pass")
confusionMatrix(pre_test1$y,predict_test,positive = "pass")
#2
svm_model2 = svm(x.train,y.train, method = "C-classification", kernel = "radial", cost = 10,
                gamma = 0.001)
summary(svm_model2)
predict_train2=predict(svm_model2,newdata = x.train)
predict_test2=predict(svm_model2,newdata = x.test)
confusionMatrix(pre_train1$y,predict_train2,positive = "pass")
confusionMatrix(pre_test1$y,predict_test2,positive = "pass")
#3
svm_model3 <- ksvm(x.train,y.train,
                 type='C-svc',kernel="rbfdot",kpar="automatic",
                 C=10, cross=5)
summary(svm_model3)
predict_train3=predict(svm_model3,newdata = x.train)
predict_test3=predict(svm_model3,newdata = x.test)
confusionMatrix(pre_train1$y,predict_train3,positive = "pass")
confusionMatrix(pre_test1$y,predict_test3,positive = "pass")

#4
svm_model4 <- ksvm(x.train,y.train,
                     type='C-svc',kernel="vanilladot", C = 10)
summary(svm_model4)
predict_train4=predict(svm_model4,newdata = x.train)
predict_test4=predict(svm_model4,newdata = x.test)
confusionMatrix(pre_train1$y,predict_train4,positive = "pass")
confusionMatrix(pre_test1$y,predict_test4,positive = "pass")

#5
tuneResult <- tune(svm, train.x = x.train, train.y = y.train, 
                   ranges = list(gamma = 10^(-4:-1),kernel="radial", cost = 2^(3:4)),class.weights= c("pass" = 1, "fail" = 10),tunecontrol=tune.control(cross=3))
print(tuneResult) 
summary(tuneResult)
tunedModel <- tuneResult$best.model;tunedModel
predict_train5=predict(tunedModel,newdata = x.train)
predict_test5=predict(tunedModel,newdata = x.test)
confusionMatrix(pre_train1$y,predict_train5,positive = "pass")
confusionMatrix(pre_test1$y,predict_test5,positive = "pass")

######  XG BOOST

pre_train1_1=pre_train1
pre_test1_1=pre_test1

y.train=as.numeric(y.train)
y.test=as.numeric(y.test)

y.train[y.train==1]=0
y.train[y.train==2]=1
y.test[y.test==1]=0
y.test[y.test==2]=1

dtrain = xgb.DMatrix(data = x.train,
                     label = y.train)
dtest = xgb.DMatrix(data = x.test,
                    label = y.test)
xg_model1 = xgboost(data = dtrain, max.depth = 4, 
                eta = 0.4, nthread = 2, nround = 40, 
                objective = "binary:logistic", verbose = 1)
watchlist = list(train=dtrain, test=dtest)
watchlist
model = xgb.train(data=dtrain, max.depth=4,
                  eta=0.3, nthread = 2, nround=20, 
                  watchlist=watchlist,
                  eval.metric = "error", 
                  objective = "binary:logistic", verbose = 1)
pred <- predict(model, as.matrix(x.test))
print(length(pred))
print(head(pred))
prediction <- ifelse(pred > 0.5, "pass", "fail")
prediction <- as.factor(prediction)
y.test[y.test==0]="fail"
y.test[y.test==1]="pass"
y.test=as.factor(y.test)
confusionMatrix(y.test, prediction,positive = "pass")

###prediction on Original test######
x.org = predict(dummies, newdata = Test_mergeddata4)
org.pred=predict(model, as.matrix(x.org))
head(org.pred)
org.prediction <- ifelse(org.pred > 0.5, "pass", "fail")
head(org.prediction)
write.csv(org.prediction, file = "XGpreds.csv",col.names = TRUE,row.names = FALSE)





######Trying with knn imputation
mergeddata3=knnImputation(mergeddata)
write.csv(mergeddata3, file = "knn_train.csv",col.names = TRUE,row.names = FALSE)
Test_mergeddata5=knnImputation(Test_mergeddata)
write.csv(Test_mergeddata5, file = "knn_test.csv",col.names = TRUE,row.names = FALSE)
mergeddata4=mergeddata3[ , -which(names(mergeddata3) %in% c("ID"))]
Test_mergeddata6=Test_mergeddata5[ , -which(names(Test_mergeddata5) %in% c("ID"))]

###Model Building
set.seed(1234)
index_train <- createDataPartition(mergeddata4$y, p = 0.7, list = F)
pre_train2 <- mergeddata4[index_train, ]
pre_test2 <- mergeddata4[-index_train, ]
dim(pre_train2)
dim(pre_test2)

#Logistic regression
log_reg2 <- glm(y ~ ., data = pre_train2, family = "binomial")
summary(log_reg2)
proab_train2 <- predict(log_reg2, type = "response")
pred2 <- prediction(proab_train2, pre_train2$y)
perf2 <- performance(pred2, measure="tpr", x.measure="fpr")
plot(perf2, col=rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))
perf_auc2 <- performance(pred2, measure="auc")
auc2 <- perf_auc2@y.values[[1]]
print(auc2)
cutoffs2 <- data.frame(cut= perf2@alpha.values[[1]], fpr= perf2@x.values[[1]], 
                       tpr=perf2@y.values[[1]])
cutoffs2 <- cutoffs2[order(cutoffs2$tpr, decreasing=TRUE),]
View(cutoffs2)
plot(perf2, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))
pred_train2 <- ifelse(proab_train2> 0.302, "pass", "fail")
table(pre_train2$y,pred_train2)
confusionMatrix(pred_train2, pre_train2$y, positive = "pass")
#on validation
prob_test2 <- predict(log_reg2, pre_test2, type = "response")
preds_test2 <- ifelse(prob_test2 > 0.302, "pass", "fail")
confusionMatrix(preds_test2, pre_test2$y, positive = "pass")
#on Original test
prob_org2 <- predict(log_reg2, Test_mergeddata6, type = "response")
preds_org2 <- ifelse(prob_org2 > 0.302, "pass", "fail")
preds_org2=data.frame(preds_org2)
write.csv(preds_org2, file = "knn_log.csv",col.names = TRUE,row.names = FALSE)


###########XGBoost#######
dummies2=dummyVars(y~.,data=pre_train2)
str(dummies2)
summary(dummies2)
x.train2=predict(dummies2, newdata = pre_train2)
y.train2=pre_train2$y
x.test2 = predict(dummies, newdata = pre_test2)
y.test2 = pre_test2$y

y.train2=as.numeric(y.train2)
y.test2=as.numeric(y.test2)

y.train2[y.train2==1]=0
y.train2[y.train2==2]=1
y.test2[y.test2==1]=0
y.test2[y.test2==2]=1

dtrain2 = xgb.DMatrix(data = x.train2,
                     label = y.train2)
dtest2 = xgb.DMatrix(data = x.test2,
                    label = y.test2)
xg_model2 = xgboost(data = dtrain2, max.depth = 4, 
                    eta = 0.4, nthread = 2, nround = 40, 
                    objective = "binary:logistic", verbose = 1)
watchlist2 = list(train=dtrain2, test=dtest2)
watchlist2
model2 = xgb.train(data=dtrain2, max.depth=4,
                  eta=0.3, nthread = 2, nround=20, 
                  watchlist=watchlist2,
                  eval.metric = "error", 
                  objective = "binary:logistic", verbose = 1)
pred2 <- predict(model2, as.matrix(x.test2))
print(length(pred2))
print(head(pred2))
prediction2 <- ifelse(pred2 > 0.5, "pass", "fail")
prediction2 <- as.factor(prediction2)
y.test2[y.test2==0]="fail"
y.test2[y.test2==1]="pass"
y.train2[y.train2==0]="fail"
y.train2[y.train2==1]="pass"
y.test2=as.factor(y.test2)
y.train2=as.factor(y.train2)
confusionMatrix(y.test2, prediction2,positive = "pass")

###prediction on Original test with knn XGBOOST######
Test_mergeddata7=Test_mergeddata6
Test_mergeddata7$y="Pass"
Test_mergeddata7$y[1]="fail"
Test_mergeddata7$y=as.factor(Test_mergeddata7$y)
Test_mergeddatay=Test_mergeddata7$y
Test_mergeddata7=Test_mergeddata7[ , -which(names(Test_mergeddata7) %in% c("y"))]
Test_mergeddata7=cbind.data.frame(Test_mergeddatay,Test_mergeddata7)
Test_mergeddata7=rename(Test_mergeddata7, y = Test_mergeddatay)
x.org2 = predict(dummies2, newdata = Test_mergeddata7)
org.pred2=predict(model2, as.matrix(x.org2))
head(org.pred2)
org.prediction2 <- ifelse(org.pred2 > 0.5, "pass", "fail")
head(org.prediction2)
write.csv(org.prediction2, file = "XGpreds_knn.csv",col.names = TRUE,row.names = FALSE)


####SVM
#6
svm_model6 = svm(x.train2,y.train2, method = "C-classification", kernel = "radial", cost = 10,
                 gamma = 0.001)
summary(svm_model6)
predict_train6=predict(svm_model6,newdata = x.train2)
predict_test6=predict(svm_model6,newdata = x.test2)
confusionMatrix(pre_train2$y,predict_train6,positive = "pass")
confusionMatrix(pre_test2$y,predict_test6,positive = "pass")
org.pred3=predict(svm_model6, x.org2)
org.pred3=data.frame(org.pred3)
write.csv(org.pred3, file = "svmpreds_knn.csv",col.names = TRUE,row.names = FALSE)

#######Cross Validation#########
set.seed(1234)
controlParameters <- trainControl(method="cv", number=6, savePredictions = TRUE, classProbs = TRUE,search = "random")
#parameterGrid <- expand.grid(mtry=c(2,3,4,5))
model_rf_cv  <- train(y~., data=mergeddata4, method="rf", trControl=controlParameters)
model_rf_cv

pred_Train = predict(model_rf_cv,mergeddata4)
confusionMatrix(mergeddata4$y,pred_Train,positive = "pass")
pred_Val = predict(model_rf_cv, Test_mergeddata6)
write.csv(pred_Val, file = "cv_preds_knn.csv",col.names = TRUE,row.names = FALSE)

###########Nueral Networks code(MLP) in other ipnb File#########
x=rbind(x.train2,x.test2)
y.train2=data.frame(y.train2)
y.test2=data.frame(y.test2)
colnames(y.train2)[colnames(y.train2)=="y.train2"] <- "y"
colnames(y.test2)[colnames(y.test2)=="y.test2"] <- "y"
y=rbind(y.train2,y.test2)
dummied_data=cbind(x,y)
write.csv(dummied_data,file = "MLP_data.csv",col.names = TRUE,row.names = FALSE)
write.csv(x.org2,file = "MLP_Test.csv",col.names = TRUE,row.names = FALSE)
MLP_result=read.csv("MLP_result.csv",header = T,sep = ",")
org.prediction3 <- ifelse(MLP_result$X0 > 0.5, "pass", "fail")
head(org.prediction3)
write.csv(org.prediction3, file = "MLP_submission.csv",col.names = TRUE,row.names = FALSE)


####CNN
CNN_result=read.csv("CNN_result.csv",header = T,sep = ",")
org.prediction4 <- ifelse(CNN_result$X0 > 0.5, "pass", "fail")
head(org.prediction4)
write.csv(org.prediction4, file = "CNN_submission.csv",col.names = TRUE,row.names = FALSE)





#####Stacking#########

all=cbind(preds_test1,pred_Test1_3,prediction,preds_test2,prediction2,predict_test6,pre_test2$y)
all_val=data.frame(all)
all=cbind(preds_org1,pred_org_Test,org.prediction,preds_org2,org.prediction2,org.pred3)
all_Test=data.frame(all)
all_Test=cbind(all_Test,o)



######################################## DATA VISUALIZATION ##################################

##Target Variable
ggplot(mergeddata3,aes(x=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Target Variable")

##merged data with target variable
ggplot(mergeddata3,aes(x=material.grade,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="material Grade Vs Target")

##
ggplot(mergeddata3,aes(x=Lubrication,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Lubrication Vs Target")

##
ggplot(mergeddata3,aes(x=Valve.Type,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Valve Type Vs Target")

##
ggplot(mergeddata3,aes(x=Bearing.Vendor,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Bearing Vendor Vs Target")

##
ggplot(mergeddata3,aes(x=Fuel.Type,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Fuel Type Vs Target")

##
ggplot(mergeddata3,aes(x=Compression.ratio,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Compression Ratio Vs Target")

##
ggplot(mergeddata3,aes(x=cam.arrangement,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="cam arrangement Vs Target")

##
ggplot(mergeddata3,aes(x=Cylinder.arragement,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Cylinder arrangement Vs Target")

##
ggplot(mergeddata3,aes(x=Turbocharger,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Turbocharger Vs Target")

##
ggplot(mergeddata3,aes(x=Varaible.Valve.Timing..VVT.,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="VVT Vs Target")

##
ggplot(mergeddata3,aes(x=Cylinder.deactivation,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Cylinder Deactivation Vs Target")

##
ggplot(mergeddata3,aes(x=Direct.injection,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Direct Injection Vs Target")

##
ggplot(mergeddata3,aes(x=main.bearing.type,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="main bearing type Vs Target")

##
ggplot(mergeddata3,aes(x=displacement,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Displacement Vs Target")

##
ggplot(mergeddata3,aes(x=piston.type,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Piston Type Vs Target")
##
ggplot(mergeddata3,aes(x=Max..Torque,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Max Torque",
       title="Max Torque Vs Target")

##
ggplot(mergeddata3,aes(x=Peak.Power,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Peak.Power Vs Target")

##
ggplot(mergeddata3,aes(x=Crankshaft.Design,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Crank Shaft Design Vs Target")

##
ggplot(mergeddata3,aes(x=Liner.Design.,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Linear Design Vs Target")

##
ggplot(mergeddata3,aes(x=a,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Test-A Vs Target")

##
ggplot(mergeddata3,aes(x=b,fill=y))+
  theme_bw()+
  geom_bar()+
  labs(y="Frequency",
       title="Test-B Vs Target")

###numeric Data###
ggplot(mergeddata3,aes(x=Number.of.Cylinders ,fill=y))+
  theme_bw()+
  geom_density(alpha=0.5)+
labs(y="Shots Frequency",
     title="Number of Cylinders Vs Target")


### Comparision within datas ######
ggplot(mergeddata3,aes(x=material.grade ,fill=y))+
  facet_wrap(~mergeddata3$Lubrication)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Material grade with lubrication")

##
ggplot(mergeddata3,aes(x=mergeddata3$Valve.Type ,fill=y))+
  facet_wrap(~mergeddata3$Lubrication)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Valve Type with lubrication")

###
ggplot(mergeddata3,aes(x=mergeddata3$Bearing.Vendor ,fill=y))+
  facet_wrap(~mergeddata3$Lubrication)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Bearing vendor with lubrication")

####
ggplot(mergeddata3,aes(x=mergeddata3$Fuel.Type ,fill=y))+
  facet_wrap(~mergeddata3$Lubrication)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Material grade with lubrication")
###
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Lubrication)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with lubrication")
###
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$material.grade)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with material Grade")
###
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Valve.Type)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Valve Type")
##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Bearing.Vendor)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Bearing Vendor")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Fuel.Type)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Fuel Type")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Compression.ratio)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Compression ratio")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$cam.arrangement)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with cam arrangement")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Cylinder.arragement)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Cylinder arrangment")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Turbocharger)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Turbocharger")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$material.grade)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with material Grade")

##
ggplot(mergeddata3,aes(x=mergeddata3$a ,fill=y))+
  facet_wrap(~mergeddata3$Liner.Design.)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-A with Linear design")

#####
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$Liner.Design.)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Linear Design")
##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$material.grade)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with material Grade")

##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$Cylinder.arragement)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Cylinder arrangements")

##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$piston.type)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Piston Type")
##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$main.bearing.type)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Main Bearing Type")
##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$Cylinder.deactivation)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Cylinder deactivation")
##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$Fuel.Type)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Fuel Type")

##
ggplot(mergeddata3,aes(x=mergeddata3$b ,fill=y))+
  facet_wrap(~mergeddata3$Valve.Type)+
  theme_bw()+
  geom_bar()+
  labs(y="Shots Frequency",
       title="Test-B with Valve Type")


####################### 3D - Scatter plots


plot3d(mergeddata3$ID,
       mergeddata3$Bearing.Vendor,
       mergeddata3$Valve.Type,
       xlab = "ID",
       ylab = "Bearing.Vendor",
       zlab = "Valve Type",
       col=brewer.pal(3,"Dark2")[unclass(mergeddata3$y)]
)

##
plot3d(mergeddata3$ID,
       mergeddata3$a,
       mergeddata3$b,
       xlab = "ID",
       ylab = "Test-A",
       zlab = "Test-B",
       col=brewer.pal(3,"Dark2")[unclass(mergeddata3$y)]
)


###
plot3d(mergeddata3$ID,
       mergeddata3$Number.of.Cylinders,
       mergeddata3$Valve.Type,
       xlab = "ID",
       ylab = "Number of Cylinders",
       zlab = "Valve Type",
       col=brewer.pal(3,"Dark2")[unclass(mergeddata3$y)]
)

###
plot3d(mergeddata3$ID,
       mergeddata3$Bearing.Vendor,
       mergeddata3$Number.of.Cylinders,
       xlab = "ID",
       ylab = "Bearing.Vendor",
       zlab = "Number of Cylinders",
       col=brewer.pal(3,"Dark2")[unclass(mergeddata3$y)]
)


###################### Results Visualizations #######################
#-----train model with cross validation--------



control <- trainControl(method="repeatedcv", number=10, repeats=3)



set.seed(7)

fit.glm <- caret::train(y~., data=mergeddata4, method="glm", trControl=control)



set.seed(7)

fit.cart <- caret::train(y~., data=mergeddata4, method="rpart", trControl=control)

set.seed(7)

fit.rf <- caret::train(y~., data=mergeddata4, method="rf", trControl=control)

set.seed(7)

fit.gbm1 <- caret::train(y~., data=mergeddata4, method="gbm", trControl=control,verbose = FALSE)

set.seed(7)

fit.xgb <- caret::train(y~., data=mergeddata4, mergeddata4="xgbTree", trControl=control,verbose = FALSE)


names(getModelInfo())

results <- resamples(list(GLM=fit.glm,CART=fit.cart, GBM=fit.gbm1, RF=fit.rf, XGB=fit.xgb))



summary(results)



#-- Model comparison Box plot for Accuracy and Kappa -----

scales <- list(x=list(relation="free"), y=list(relation="free"))

bwplot(results, scales=scales)
