#Analysis to Predict Credit Card Fraud Data, with anonymized credit card transactions labeled as fraudulent or genuine


setwd('C:/Users/m339673/Desktop/CreditCard')

## load libraries
library(randomForest)
library(e1071)
library(rpart)
library(rpart.plot)
library(caTools)
library(caret)
library(DMwR)
library(ROCR)
library(class)
library(ggplot2)
library(mgcv)

#getting input
card <- read.csv("creditcard.csv")
summary(card)
head(card[,23:29])


#Modify some of the variables
##convert class variable to factor
card$Class<-factor(card$Class)
card$Amount_scaled<-scale(card$Amount)
cardcenter<-attr(card$Amount_scaled, "scaled:center")
cardscale<-attr(card$Amount_scaled, "scaled:scale")

# check output Class distribution for baseline accuracy
table(card$Class)[1]/dim(card)[1]  #.9982725, data is highly imbalanced, so accuracy will not be a good measure of model performance.

#Variables to use for modeling(Removing Time and Amount)
vars.to.use<-colnames(card)[c(-1,-30)]
data<-as.data.frame(card[,vars.to.use])
numericVars<-colnames(data[c(-29,-31)])
outcome<-'Class'
pos<-'1'

#Sample data for modeling
set.seed(1234)
data$gp<-runif(dim(data)[1])
test<-subset(data, data$gp<=.3) #test data
train<-subset(data, data$gp>.3) #initial training before split

# check output Class distribution for baseline accuracy
table(train$Class)[1]/dim(train)[1]  #.998224
table(test$Class)[1]/dim(test)[1]  #.9983857


#Balance training data using SMOTE to balance classes
newcard<-SMOTE(Class ~ .,train,perc.over=10000, perc.under=101)
table(newcard$Class)

#Split training data into training and calibration
useforCal<-rbinom(n=dim(newcard)[[1]],size=1, prob=0.1)>0 #index rows to be used for calibration
Cal<-subset(newcard,useforCal) #calibration data
credtrain<-subset(newcard,!useforCal) #training data
dim(credtrain)[1]
dim(Cal)[1]


# check output Class distribution for new training and calibration sets
table(credtrain$Class)[1]/dim(credtrain)[1]  #.5012288
table(Cal$Class)[1]/dim(Cal)[1]  #.4890582

#Start with building single variable models
calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

mkPredC <- function(outCol,varCol,fraudCol) { 
  pPos <- sum(outCol==pos)/length(outCol) 
  naTab <- table(as.factor(outCol[is.na(varCol)]))
  pPosWna <- (naTab/sum(naTab))[as.character(pos)] 
  vTab <- table(as.factor(outCol),varCol)
  pPosWv <- (vTab[as.character(pos),]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3)
  pred <- pPosWv[fraudCol]
  pred[is.na(fraudCol)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred 
}

mkPredN<-function(outCol, varCol,fraudCol) {
  cuts<-unique(as.numeric(quantile(varCol,
    probs=seq(0,1,0.1),na.rm=T)))
  varC<-cut(varCol,cuts)
  fraudC<-cut(fraudCol,cuts)
  mkPredC(outCol,varC,fraudC)
}


for(v in numericVars) {
  pi <- paste('pred',v,sep='')
  credtrain[,pi] <- mkPredN(credtrain[,outcome],credtrain[,v],credtrain[,v])
  test[,pi] <- mkPredN(credtrain[,outcome],credtrain[,v],test[,v])
  Cal[,pi] <- mkPredN(credtrain[,outcome],credtrain[,v],Cal[,v])
  aucTrain <- calcAUC(credtrain[,pi],credtrain[,outcome])
  if(aucTrain>=0.85) {
    aucCal <- calcAUC(Cal[,pi],Cal[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f",
                  pi,aucTrain,aucCal))
  }
}


#Cross validate variables
fCross <- function() {
  useForCalRep <- rbinom(n=dim(credtrain)[[1]],size=1,prob=0.1)>0
  predRep <- mkPredN(credtrain[!useForCalRep,outcome],
                     credtrain[!useForCalRep,var],
                     credtrain[useForCalRep,var])
  calcAUC(predRep,credtrain[useForCalRep,outcome])
}


for(v in numericVars) {
  var <- paste('pred',v,sep='')
  aucs <- replicate(100,fCross())
  if(mean(aucs)>=0.90) {
    meanaucs<-mean(aucs)
    aucTrain <- calcAUC(credtrain[,var],credtrain[,outcome])
    aucCal <- calcAUC(Cal[,var],Cal[,outcome])
    aucTest<- calcAUC(test[,var],test[,outcome])
    print(sprintf("%s, trainAUC: %4.3f calibrationAUC: %4.3f testAUC:%4.3f  meanXAUC: %4.3f",
                  var,aucTrain,aucCal, aucTest,meanaucs))
  }
}

#Plot a double density plot to see if best single variable model is predictive
ggplot(data=Cal)+
  geom_density(aes(x=predV4, color=as.factor(Class)))

#plot ROC curve to find optimal cutoff
eval<-prediction(test$predV17,test$Class)
plot(performance(eval,"tpr","fpr"))
cutoff<-attributes(performance(eval,'auc'))$y.values[[1]] #optimal value for variable is .9725706


cM<-table(truth=test$Class, prediction=test$predV17>=cutoff)
cM

# Attempt to build a decision tree
## Variable selection by picking variables with deviance improvement on Calibration dataset

logLikelyhood <- function(outCol,predCol) {
  sum(ifelse(outCol==pos,log(predCol),log(1-predCol)))
}

selVars <- c()
minStep <- 5
baseRateCheck <- logLikelyhood(Cal[,outcome],
                               sum(Cal[,outcome]==pos)/length(Cal[,outcome]))

for(v in numericVars) { 
  pi <- paste('pred',v,sep='')
  liCheck <- 2*((logLikelyhood(Cal[,outcome],Cal[,pi]) -
                   baseRateCheck))
  if(liCheck>=minStep) {
    print(sprintf("%s, calibrationScore: %g",
                  pi,liCheck))
    selVars <- c(selVars,pi)
  }
}

f <- paste(outcome,'==1 ~ ',paste(selVars,collapse=' + '),sep='')
f
tmodel <- rpart(f,data=credtrain,
                control=rpart.control(cp=0.001,minsplit=1000,
                                      minbucket=1000,maxdepth=5)
)

credtrain$tpred<-predict(tmodel,newdata=credtrain)
Cal$tpred<-predict(tmodel,newdata=Cal)
test$tpred<-predict(tmodel,newdata=test)

calcAUC(credtrain$tpred,credtrain[,outcome])
calcAUC(Cal$tpred,Cal[,outcome])
calcAUC(test$tpred,test[,outcome])  #AUC=0.98065


prp(tmodel) 


#Now lets build a nearest neighbor model
nK<-250 #Number of neighbors analyzed
knnTrain<-credtrain[,selVars] #variables used for classification
knnCl<-credtrain[,outcome]==pos #training outcomes


knnPred <- function(df) {
  knnDecision <- knn(knnTrain,df,knnCl,k=nK,prob=T)
  ifelse(knnDecision==TRUE,
         attributes(knnDecision)$prob,
         1-(attributes(knnDecision)$prob))
}

credtrain$nnpred<-knnPred(credtrain[,selVars])
Cal$nnpred<-knnPred(Cal[,selVars])
test$nnpred<-knnPred(test[,selVars])


calcAUC(credtrain$nnpred,credtrain[,outcome])
calcAUC(Cal$nnpred,Cal[,outcome])
calcAUC(test$nnpred,test[,outcome])  #AUC=0.98768

#Try Naive Bayes
ff <- paste('as.factor(',outcome,'==1) ~ ',
            paste(selVars,collapse=' + '),sep='')


nbmodel<-naiveBayes(as.formula(ff), data=credtrain)
credtrain$nbpred<-predict(nbmodel,newdata=credtrain,type='raw')[,'TRUE']
Cal$nbpred<-predict(nbmodel,newdata=Cal,type='raw')[,'TRUE']
test$nbpred<-predict(nbmodel,newdata=test,type='raw')[,'TRUE']

calcAUC(credtrain$nbpred,credtrain[,outcome])
calcAUC(Cal$nbpred,Cal[,outcome])
calcAUC(test$nbpred,test[,outcome]) #AUC=.98449


# Try logistic regression
fm <- paste(outcome,'==1 ~ ',paste(numericVars,collapse=' + '),sep='')
fm
model<-glm(fm, data=credtrain, family=binomial(link="logit"))
credtrain$logpred<-predict(model, newdata=credtrain, type="response")
Cal$logpred<-predict(model, newdata=Cal, type="response")
test$logpred<-predict(model,newdata=test, type="response")

calcAUC(credtrain$logpred,credtrain[,outcome])
calcAUC(Cal$logpred,Cal[,outcome])
calcAUC(test$logpred,test[,outcome]) #AUC=.9888959

## plot distribution of prediction scores
ggplot(credtrain, aes(x=logpred, color=Class, linetype=Class))+
  geom_density()

coefficients(model)
summary(model)


# Try stepwise
fm2<-paste(outcome,'==1 ~ ',1,sep='')
nothing <- glm(fm2,data=credtrain,family=binomial)

step.model<-step(nothing, direction = "both", trace = 1, scope = fm)
summary(step.model)
credtrain$slogpred<-predict(step.model, newdata=credtrain, type="response")
Cal$slogpred<-predict(step.model, newdata=Cal, type="response")
test$slogpred<-predict(step.model,newdata=test, type="response")

calcAUC(credtrain$slogpred,credtrain[,outcome])
calcAUC(Cal$slogpred,Cal[,outcome])
calcAUC(test$slogpred,test[,outcome]) #AUC=.9887558


# Try Random Forest
fmodel<-randomForest(x=credtrain[,numericVars],
                     y=credtrain$Class,
                     ntree=100,
                     nodesize=7,
                     importance=T)


#Find most important variables to rerun the model
varImp<-importance(fmodel)
varImp[1:10,]
varImpPlot(fmodel,type=1)
selVars<-names(sort(varImp[,1], decreasing=T))[1:15]

fmodel<-randomForest(x=credtrain[,selVars],
                     y=credtrain$Class,
                     ntree=1000,
                     nodesize=7,
                     importance=T)


fresults<-predict(fmodel,newdata=credtrain, "prob")
fresultsCal<-predict(fmodel,newdata=Cal, "prob")
fresultstest<-predict(fmodel,newdata=test, "prob")

credtrain$forestpred<-fresults[,2]
Cal$forestpred<-fresultsCal[,2]
test$forestpred<-fresultstest[,2]

calcAUC(credtrain$forestpred,credtrain[,outcome])
calcAUC(Cal$forestpred,Cal[,outcome])
calcAUC(test$forestpred,test[,outcome])  #AUC: .992192



#Try explicit Kernel transormation
phi <- function(x) { 
  x <- as.numeric(x)
  c(x,x*x,combn(x,2,FUN=prod))
}


phiNames <- function(n) { 
  c(n,paste(n,n,sep=':'),
    combn(n,2,FUN=function(x) {paste(x,collapse=':')}))
}


fmm <- paste('~ 0 +',paste(selVars,collapse=' + '),sep='')
modelMatrix <- model.matrix(as.formula(fmm),credtrain)
colnames(modelMatrix) <- gsub('[^a-zA-Z0-9]+','_',
                              colnames(modelMatrix))


pM <- t(apply(modelMatrix,1,phi))
vars <- phiNames(colnames(modelMatrix))
vars <- gsub('[^a-zA-Z0-9]+','_',vars)
colnames(pM) <- vars


pM <- as.data.frame(pM)
pM$Class <- credtrain$Class


#Modeling using the explicit kernel transform
formulaStr2 <- paste(outcome,'==1 ~ ',paste(vars,collapse=' + '),sep='')

m2 <- lm(as.formula(formulaStr2),data=pM)


coef2 <- summary(m2)$coefficients

#Significant Variables
SigVars <- setdiff(rownames(coef2)[abs(coef2[,'t value'])>1],
                           '(Intercept)')
SigVars <- union(colnames(modelMatrix),SigVars) 


formulaStr3 <- paste(outcome,'==1 ~ ',paste(SigVars,collapse=' + '),sep='')

modelK<-glm(formulaStr3, data=pM, family=binomial(link="logit"))

summary(modelK)

coef3 <- summary(modelK)$coefficients

#Significant Variables
SigVars <- setdiff(rownames(coef3)[coef3[,'Pr(>|z|)']<0.01],
                   '(Intercept)')

formulaStr4 <- paste(outcome,'==1 ~ ',paste(SigVars,collapse=' + '),sep='')

modelK<-glm(formulaStr4, data=pM, family=binomial(link="logit"))

summary(modelK)


### Create test set with same variables
fmm <- paste('~ 0 +',paste(selVars,collapse=' + '),sep='')
modelMatrixT <- model.matrix(as.formula(fmm),test)
colnames(modelMatrixT) <- gsub('[^a-zA-Z0-9]+','_',
                              colnames(modelMatrixT))


pMtest <- t(apply(modelMatrixT,1,phi))
varstest <- phiNames(colnames(modelMatrixT))
varstest <- gsub('[^a-zA-Z0-9]+','_',varstest)
colnames(pMtest) <- varstest


pMtest <- as.data.frame(pMtest)
pMtest$Class <- test$Class




pM$klogpred<-predict(modelK, newdata=pM, type="response")
pMtest$klogpred<-predict(modelK,newdata=pMtest, type="response")

calcAUC(pM$klogpred,pM[,outcome])
calcAUC(Cal$logpred,Cal[,outcome])
calcAUC(pMtest$klogpred,pMtest[,outcome]) #AUC=0.9862059










#Function to measure accuracy, F1 and deviance to compare models
loglikelihood <- function(y, py) {   	# Note: 3 
  pysmooth <- ifelse(py==0, 1e-12,
                     ifelse(py==1, 1-1e-12, py))
  
  sum(y * log(pysmooth) + (1-y)*log(1 - pysmooth))
}


accuracyMeasures <- function(pred, truth, name="model") { 
  dev.norm <- -2*loglikelihood(as.numeric(truth), pred)/length(pred)
  ctable <- table(truth=truth,
                  pred=(pred>0.5))                            
  accuracy <- sum(diag(ctable))/sum(ctable)
  precision <- ctable[2,2]/sum(ctable[,2])
  recall <- ctable[2,2]/sum(ctable[2,])
  f1 <- 2*precision*recall/(precision+recall)
  data.frame(model=name, accuracy=accuracy, f1=f1, dev.norm)
}
## Decision Tree
accuracyMeasures(test$tpred, test$Class==1, name='decision tree, test')
##Naive Bayes
accuracyMeasures(test$nbpred, test$Class==1, name='Naive Bayes, test')
##Nearest Neighbor
accuracyMeasures(test$nnpred, test$Class==1, name='Nearest Neighbor, test')
##Logistic Regression
accuracyMeasures(test$logpred, test$Class==1, name='Logistic Regression, test')
##Random Forest
accuracyMeasures(test$forestpred, test$Class==1, name='Random Forest, test')
##Kernel Regression
accuracyMeasures(pMtest$klogpred, pMtest$Class==1, name='kernel regression, test')



ggplot(credtrain, aes(x=forestpred, color=Class, linetype=Class))+
  geom_density()
