task1.Rmd
================

## The Spark Foundation “GRIP JUNE 2021”

## Task 1:- Prediction Using Supervised ML

### By:- Aakriti Pankaj

**Objective:- To predict percentage of a student if he/she study for
9.25 hrs/day.**

**Solution:-**

Read the data from url. Here is the code Chunk for reading url:-

``` r
data <- read.csv(url("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"))
data
```

    ##    Hours Scores
    ## 1    2.5     21
    ## 2    5.1     47
    ## 3    3.2     27
    ## 4    8.5     75
    ## 5    3.5     30
    ## 6    1.5     20
    ## 7    9.2     88
    ## 8    5.5     60
    ## 9    8.3     81
    ## 10   2.7     25
    ## 11   7.7     85
    ## 12   5.9     62
    ## 13   4.5     41
    ## 14   3.3     42
    ## 15   1.1     17
    ## 16   8.9     95
    ## 17   2.5     30
    ## 18   1.9     24
    ## 19   6.1     67
    ## 20   7.4     69
    ## 21   2.7     30
    ## 22   4.8     54
    ## 23   3.8     35
    ## 24   6.9     76
    ## 25   7.8     86

Plot the relationship between Hours and Scores. Here is the code chunk:-

``` r
plot(data$Hours, data$Scores, col = "blue", pch = 3, xlab = "Hours", ylab = "Scores")
title(main = "Graph 1:- Relationship between Hours and Scores")
```

![](tsk1_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

From graph 1, we can see that there is positive relationship between
Hours and Scores.

Partiton the data into training set and test set. Here is the code
chunk:-

``` r
library(lattice)
library(ggplot2)
library(caret)
set.seed(22)
inTrain <- createDataPartition(y = data$Scores, p = 0.6, list = FALSE)
traindata <- data[inTrain,]
testdata <- data[-inTrain,]
traindata
```

    ##    Hours Scores
    ## 1    2.5     21
    ## 2    5.1     47
    ## 3    3.2     27
    ## 6    1.5     20
    ## 7    9.2     88
    ## 8    5.5     60
    ## 10   2.7     25
    ## 12   5.9     62
    ## 14   3.3     42
    ## 15   1.1     17
    ## 16   8.9     95
    ## 19   6.1     67
    ## 21   2.7     30
    ## 22   4.8     54
    ## 23   3.8     35
    ## 24   6.9     76
    ## 25   7.8     86

Fit a linear model. Here is the code chunk:-

``` r
lm1 <- lm(Scores ~ Hours, data = traindata)
summary(lm1)
```

    ## 
    ## Call:
    ## lm(formula = Scores ~ Hours, data = traindata)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -7.675 -5.209  2.330  3.520  6.927 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)   1.1769     2.7216   0.432    0.672    
    ## Hours        10.2715     0.5103  20.127 2.88e-12 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 5.04 on 15 degrees of freedom
    ## Multiple R-squared:  0.9643, Adjusted R-squared:  0.9619 
    ## F-statistic: 405.1 on 1 and 15 DF,  p-value: 2.879e-12

Plot prediction of training set and test set with best line of fit. Here
is the code chunk:-

``` r
plot(traindata$Hours, traindata$Scores, pch = 3, col = "blue", xlab = "Hours", ylab = "Scores")
lines(traindata$Hours, predict(lm1), lwd = 3)
title(main = "Graph 2:- Relationship between Hours and Scores of training set")
```

![](tsk1_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
plot(testdata$Hours, testdata$Scores, pch = 3, col = "blue", xlab = "Hours", ylab = "Scores")
lines(testdata$Hours, predict(lm1, newdata = testdata), lwd = 3)
title(main = "Graph 3:- Relationship between Hours and Scores of Test set")
```

![](tsk1_files/figure-gfm/unnamed-chunk-5-2.png)<!-- -->

From graph 2 and 3, we can see that there is linear relationship between
Hours and Scores of both training and test data set.

Calculate training set error. Here is the code chunk:-

``` r
sqrt(sum((lm1$fitted - traindata$Scores)^2))
```

    ## [1] 19.52038

Calculate test set error. Here is the code chunk:-

``` r
sqrt(sum((predict(lm1, newdata = testdata) - testdata$Scores)^2))
```

    ## [1] 20.32981

Predict score if a student study for 9.25 hrs/ day. Here is the code
chunk:-

``` r
newdata <- data.frame(Hours = 9.25)
predict(lm1, newdata)
```

    ##        1 
    ## 96.18837

As a result, we get 96.18837% score.

**Conclusion:-** Using Supervised ML model, predicted percentage of a
student is 96.18837% if he/she study for 9.25 hrs/day.
