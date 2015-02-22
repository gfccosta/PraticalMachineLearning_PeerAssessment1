# Coursera Course: Pratical Machine Learning - Course Project
Genaro Costa  
February, 19 2015  

The act of logging the activity some one perform on gim is a boring task. To tackle with that problem this report will use aceleromenter data, colected and provided by [1], to predict the user activity. The autor in [1] collected information from acelerometers placed on arm, forearm, belt and dumbbel among different users performing five different activities.. In this work we experiment such information to traing a prediction model and to try to predict which activity an user is performing based on its accelerometer data.

The code bellow setup the requirements libraries and configure the system for parallel performance.

```r
    library(lattice)
    library(ggplot2)
    library(caret)
    library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
    library(RANN)
    library(plyr)
    library(ipred)
    library(doMC)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
    ## set the locale to English to better deal with conversions.
    lastLocale <- Sys.setlocale(category = "LC_ALL", locale = "en_US.UTF-8")
    
    ## confire for multicore use
    registerDoMC(cores = 8)

    ## set the seed for reproducibility
    set.seed(13353)
```

##Data preprocessing


```r
    ## loads the data
    data <- read.csv(file = 'pml-training.csv', na.strings=c("","#DIV/0!","NA"))

    ## show some data dimensions
    dim(data)
```

```
## [1] 19622   160
```

```r
    ## remove time and user related columns 
    cleanData <- data[,-which(names(data) %in% c("X","user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp"))]
    
    ## convert 'integer' and 'logic' types to 'numeric'
    for(i in seq(ncol(cleanData))) { 
            if(class(cleanData[1,i])=='integer' || class(cleanData[1,i])=='logical') {
                    cleanData[,i] <- as.numeric(cleanData[,i])
            }
    }
    ## remove near zero variability columns
    nsv <- nearZeroVar(cleanData)
    varData <- cleanData[,-nsv]
```

We loaded the data, considering the values ‘NA’ and ‘#DIV/0!’ as *NA* (missing values). The data has 160 columns. The columns related to time and user names where removed from the data. And all columns types different from *numeric* where converted to *numeric*. We also removed the near zero variability variables from the dataset. That reduces the variables to 119.

##Machine learning model generation

The machine learning algotithms used are provided by the *caret* R package. The data were divided in two different sets for cross-validation. The training and testing sets has 70% and 30% respectively. 


```r
    inTrain <- createDataPartition(varData$classe, p=.7, list = FALSE)
    training <- varData[inTrain,]
    testing <- varData[-inTrain,]
```

We first use the Random Forest algorithm in the training phase. As we have a lot of missing values, we use the knnInpute algorithm to fill the values missing.


```r
    ## if there is a cached model, loads it
    if(file.exists('models.bin')) { 
        load('models.bin')
    }
    if(!exists('modFit')) {
        ## train only if the models doesnot exists.
        modFit <- train(classe ~ ., data=training, method="rf", preProcess = c("knnImpute"), na.action  = na.pass)
    }

    ## performs crossvalidation
    prediction <- predict(modFit, testing, na.action  = na.pass)

    ## get the confusion matrix
    confusionMatrix(prediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    2    0    0    0
##          B    0 1136    3    0    0
##          C    0    1 1023    7    0
##          D    0    0    0  957    4
##          E    1    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9969          
##                  95% CI : (0.9952, 0.9982)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9961          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9971   0.9927   0.9963
## Specificity            0.9995   0.9994   0.9984   0.9992   0.9998
## Pos Pred Value         0.9988   0.9974   0.9922   0.9958   0.9991
## Neg Pred Value         0.9998   0.9994   0.9994   0.9986   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1930   0.1738   0.1626   0.1832
## Detection Prevalence   0.2846   0.1935   0.1752   0.1633   0.1833
## Balanced Accuracy      0.9995   0.9984   0.9977   0.9960   0.9980
```

```r
    ## calculate the accuracy.
    accuracy <- sum(prediction==testing$classe)/nrow(testing)
```

As presented in the confusion matrix above, we have a good accuracy (99.69%). To find out if we could get a better accuracy, we used bagging on random forest algorithm. In that scenarion we have an **out of sample error** of 0.31%.


```r
    if(!exists('modFit2')) {
        ## train only if the models doesnot exists.
        modFit2 <- train(classe ~ ., data=training, method="treebag", preProcess = c("knnImpute"), na.action  = na.pass)
    }
    
    ## performs crossvalidation
    prediction2 <- predict(modFit2, testing, na.action  = na.pass)
    
    ## get the confusion matrix
    confusionMatrix(prediction2, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    5    0    0    0
##          B    1 1132    2    1    0
##          C    0    2 1024    3    0
##          D    0    0    0  960    3
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9971          
##                  95% CI : (0.9954, 0.9983)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9963          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9939   0.9981   0.9959   0.9972
## Specificity            0.9988   0.9992   0.9990   0.9994   1.0000
## Pos Pred Value         0.9970   0.9965   0.9951   0.9969   1.0000
## Neg Pred Value         0.9998   0.9985   0.9996   0.9992   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1924   0.1740   0.1631   0.1833
## Detection Prevalence   0.2851   0.1930   0.1749   0.1636   0.1833
## Balanced Accuracy      0.9991   0.9965   0.9985   0.9976   0.9986
```

```r
    ## calculate the accuracy.
    accuracy2 <- sum(prediction2==testing$classe)/nrow(testing)
```

As presented by the confusion matrix above, when using bagging technique, we got a little better accuracy (99.71%) than the accuracy (99.69%) achieved when using only the random forest algorithm. Using bagging we have an **out of sample error** of 0.29%. 

##References
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
