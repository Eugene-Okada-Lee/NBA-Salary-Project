#This project below was done for my one of my

#This code is used for the analysis
rm(list = ls()) #clear list


library(gamlr)
library(tidyverse)
library(ggpubr)
library(rstatix)


#This is to import the dataset scraped from the internet NBA.com 
NBA <- read.csv("C:\\Users\\eugen\\Documents\\NBA Salary Predicitons\\NBA_Salary_Info_2018-19.csv")

#View(NBA)

changeFactor <- c("Position", "Team", "Nationality","Draft.Round", "Draft.number","Draft.Year") # create a vector with all variables that need to be changed to factors
NBA[changeFactor] <- lapply(NBA[changeFactor], factor) #Change catagroical data to factor

is.na(NBA) #This is to check if there are any players that are missing data points
NBA <- na.omit(NBA) #this is to omit any players who are missing data points

sum(NBA$Player)

####################
#Summary Statistics#
####################

n <- length(NBA$Player) #total number of players in observation
avg_salary <- mean(NBA$SALARY)
hist(NBA$SALARY, col = 'skyblue', main = "Histogram of NBA Salary", ylab = "Number of Players", xlab = "Salary $$", las =1)
abline(v= avg_salary, col = "red")


over_avg <- sum(NBA$SALARY > avg_salary) #Number of players making above the average salary
under_avg <- sum(NBA$SALARY <= avg_salary) #Number of players making below the average salary

cat( over_avg, " of the 465 players have a salary above the average amount of $", avg_salary)
cat( under_avg, " of the 465 players have a salary below the average amount of $", avg_salary)
cat("this means that", (over_avg/n)*100  ,"% of players make a majority of the salary distributed throughout the NBA")
cat("this means that", (1-over_avg/n)*100, "% of players make less than the average amount of NBA money")
#this shows that the majority of players make under the mean of  

#Now I create a subset data set for international players and US players, and include summary statistics 
US <- NBA[NBA$Nationality == "USA",]
mean(US$SALARY)

#histogran which plots US player salary with a line denoting the average of all NBA average
hist(US$SALARY, col = 'skyblue', main = "Histogram of NBA Salary", ylab = "Number of Players", xlab = "Salary $$", las =1)
abline(v= avg_salary, col = "red")

Non_US <- NBA[NBA$Nationality != "USA",]
mean(Non_US$SALARY)

#histogran which plots US player salary with a line denoting the average of all NBA average
hist(Non_US$SALARY, col = 'skyblue', main = "Histogram of NBA Salary", ylab = "Number of Players", xlab = "Salary $$", las =1)
abline(v= avg_salary, col = "red")

#our two groups look very similar, which is what we want
#Summary to see the effect of basketball stats on salary
coef(summary(glm(SALARY ~ PPG + RPG + APG + Net.Rating, data=NBA)))

#International 
coef(summary(glm(SALARY ~ PPG + RPG + APG + Net.Rating, data=Non_US)))

#US
coef(summary(glm(SALARY ~ PPG + RPG + APG + Net.Rating, data=US)))


#Below create balance tables to check on the mean of PPG,RPG,ASG, based on nationality
NBA%>%
  group_by(Nationality) %>%
  get_summary_stats(PPG, type = "mean_sd")

NBA%>%
  group_by(Nationality) %>%
  get_summary_stats(RPG, type = "mean_sd")


NBA%>%
  group_by(Nationality) %>%
  get_summary_stats(APG, type = "mean_sd")



################
#MODEL BUILDING#
################
#below is the code to create 3 models, one for the entirety of the NBA, one for US players, and one for International players
#and then the two models will be used to project the same players salary based on the stats to see if there is a huge descrepency in salary based on nationality
#Trying to project the future salaries of US born players and then compare them to the international players 


n1 <- length(US$Player)
n2 <- length(Non_US$Player)
#develope folds so that the data is split into 10 percents

#Set random seed to keep models consistent
set.seed(1234)
#Model Building using statistics from the entire NBA 
K <- 10 #this is so that we can divide the data set into 10 different groups so that we can assign 1 group which will make 10% of the dataset into the test set
folds <- rep(1:K,each=ceiling(n/K))
randomized_NBA <- sample(1:n)
random_folds <- folds[randomized_NBA]
training_set_indices <- random_folds!=1

training_set <- NBA[training_set_indices,] #random traing set with 90% of the US players
test_set <-  NBA[!training_set_indices,] #random test set with 10% of the US players


XNBA <- model.matrix(~., data=training_set[,!(names(training_set) %in% c("Player","College","SALARY"))])[,-1]
dim(XNBA)
YNBA <- training_set$SALARY
lasso <- gamlr(XNBA, log(YNBA))
lasso
plot(lasso, main="Regularization Path for Log Salary", ylab="estimated betas")


#Find the optimal model
cv.lasso_model <- cv.gamlr(XNBA, log(YNBA), nfold =10, verb=TRUE)
plot(cv.lasso_model, main="OOS error estimates plot")

#
extracted_coeff <- coef(cv.lasso_model, select="min")
drop(extracted_coeff)

#
AICc_Betas <- coef(lasso)[-1,]
drop(AICc_Betas[which(AICc_Betas!=0)])


#In sample fit of the optimal lasso model
mean_y <- mean(log(training_set$SALARY))
y <- log(training_set$SALARY)
training_pred <- predict(lasso, newdata = XNBA, type="response")

IS_R2 <- 1 - (sum((y-training_pred)^2))/sum((y-mean_y)^2) 
IS_R2
#Out of Sample fit of the the optimal lasso model

new_y <- log(test_set$SALARY)
XNBA1 <- model.matrix(~., data=test_set[,!(names(test_set) %in% c("Player","College","SALARY"))])[,-1]
test_pred <- predict(lasso, newdata = XNBA1, type="response")

OOS_R2 <- 1 - (sum((new_y-test_pred)^2))/sum(new_y-mean_y)^2
OOS_R2


#Model Building using statistics from US Players only
K <- 10 #this is so that we can divide the data set into 10 different groups so that we can assign 1 group which will make 10% of the dataset into the test set
folds <- rep(1:K,each=ceiling(n1/K))
randomized_US <- sample(1:n1)
random_folds <- folds[randomized_US]
training_set_indices <- random_folds!=1

training_set_US <- US[training_set_indices,] #random traing set with 90% of the US players
test_set_US <-  US[!training_set_indices,] #random test set with 10% of the US players


X <- model.matrix(~., data=training_set_US[,!(names(training_set_US) %in% c("Player","College","SALARY"))])[,-1]
dim(X)
Y <- training_set_US$SALARY
lasso_US <- gamlr(X, log(Y))
lasso_US
plot(lasso_US, main="Regularization Path for Log Salary", ylab="estimated betas")


#Find the optimal model
cv.lasso_model_US <- cv.gamlr(X, log(Y), nfold =10, verb=TRUE)
plot(cv.lasso_model, main="OOS error estimates plot")

#
extracted_coeff <- coef(cv.lasso_model_US, select="min")
drop(extracted_coeff)

#
AICc_Betas <- coef(lasso_US)[-1,]
drop(AICc_Betas[which(AICc_Betas!=0)])


#In sample fit of the optimal lasso model
mean_y <- mean(log(training_set_US$SALARY))
y <- log(training_set_US$SALARY)
training_pred <- predict(lasso_US, newdata = X, type="response")

IS_R2_US <- 1 - (sum((y-training_pred)^2))/sum((y-mean_y)^2) 
IS_R2_US
#Out of Sample fit of the the optimal lasso model

new_y <- log(test_set_US$SALARY)
X1 <- model.matrix(~., data=test_set_US[,!(names(test_set_US) %in% c("Player","College","SALARY"))])[,-1]
test_pred <- predict(lasso_US, newdata = X1, type="response")

OOS_R2_US <- 1 - (sum((new_y-test_pred)^2))/sum(new_y-mean_y)^2
OOS_R2_US


#Model Building using statistics from International Players only
K <- 10 #this is so that we can divide the data set into 10 different groups so that we can assign 1 group which will make 10% of the dataset into the test set
folds <- rep(1:K,each=ceiling(n2/K))
randomized_Int <- sample(1:n2)
random_folds <- folds[randomized_Int]
training_set_indices <- random_folds!=1

training_set_Int <- Non_US[training_set_indices,] #random traing set with 90% of the US players
test_set_Int <-  Non_US[!training_set_indices,] #random test set with 10% of the US players


X2 <- model.matrix(~., data=training_set_Int[,!(names(training_set_Int) %in% c("Player","College","SALARY"))])[,-1]
dim(X2)
Y2 <- training_set_Int$SALARY
lasso_Int <- gamlr(X2, log(Y2))
lasso_Int
plot(lasso_Int, main="Regularization Path for Log Salary", ylab="estimated betas")


#Find the optimal model
cv.lasso_model_Int <- cv.gamlr(X, log(Y), nfold =10, verb=TRUE)
plot(cv.lasso_model_Int, main="OOS error estimates plot")

#
extracted_coeff <- coef(cv.lasso_model_Int, select="min")
drop(extracted_coeff)

#
AICc_Betas <- coef(lasso_Int)[-1,]
drop(AICc_Betas[which(AICc_Betas!=0)])


#In sample fit of the optimal lasso model
mean_y <- mean(log(training_set_Int$SALARY))
y <- log(training_set_Int$SALARY)
training_pred <- predict(lasso_Int, newdata = X2, type="response")

IS_R2_Int <- 1 - (sum((y-training_pred)^2))/sum((y-mean_y)^2) 
IS_R2_Int
#Out of Sample fit of the the optimal lasso model

new_y <- log(test_set_Int$SALARY)
X3 <- model.matrix(~., data=test_set_Int[,!(names(test_set_Int) %in% c("Player","College","SALARY"))])[,-1]
test_pred <- predict(lasso_Int, newdata = X3, type="response")

OOS_R2_Int <- 1 - (sum((new_y-test_pred)^2))/sum(new_y-mean_y)^2
OOS_R2_Int


#############
#Predictions#
#############


#All NBA Model
Salary_Prediction_NBA <- predict(lasso, newdata=XNBA, type = "response")
Lebron_Salary_NBA_Model<-Salary_Prediction_NBA[which(training_set$Player=="LeBron James")]
exp(Lebron_Salary_NBA_Model) #Get the total salary in non log form

#US Model
Salary_Prediction_US <- predict(lasso_US, newdata=XNBA, type = "response")
Lebron_Salary_US_Model<-Salary_Prediction_US[which(training_set$Player=="LeBron James")]
exp(Lebron_Salary_US_Model) #Get the total salary in non log form

#International Model
Salary_Prediction_Int <- predict(lasso_Int, newdata=XNBA, type = "response")
Lebron_Salary_Int_Model<-Salary_Prediction_Int[which(training_set$Player=="LeBron James")]
exp(Lebron_Salary_Int_Model) #Get the total salary in non log form






################
## EXTRA WORK ##
################




#Interaction on Team

X2 <- model.matrix(~. *Team, data=training_set[,!(names(training_set) %in% c("Player","SALARY","College"))])[,-1]
lasso_interaction <- gamlr(X2, log(training_set$SALARY))
lasso_interaction

#Part B
dim(X2)
#IS R2
training_pred_new <- predict(lasso_interaction, newdata = X2, type="response")
IS_R2_New <- 1 - (sum((y-training_pred_new)^2))/sum((y-mean_y)^2) 

#Out of Sample R^2


X3 <- model.matrix(~.*Team, data=test_set[,!(names(training_set) %in% c("Player","College","SALARY"))])[,-1]
test_pred_new <- predict(lasso_interaction, newdata = X3, type="response")

OOS_R2_new <- 1 - (sum((new_y-test_pred_new)^2))/sum(new_y-mean_y)^2










OLS_Train <- glm(log(SALARY) ~ ., data=training_set_US[,!(names(training_set_US) %in% c("Player","Nationality","College"))])#Run OLS with all covariates except player name
summary(OLS_Train)
#BOS, UTAH, TS, WS.48, 


#TRB.             0.9668834  0.7842618   1.233 0.218667    
#AST.             0.0298604  0.0145812   2.048 0.041510 *  
#STL.            -0.1821170  0.1371958  -1.327 0.185454    
#BLK.            -0.1737069  0.1047338  -1.659 0.098329 .  
#Below is the code for In Sample R^2 with a statement
cat("The In Sample R^2 is ", 1 - OLS_Train$deviance / OLS_Train$null.deviance, ".", sep="")



#Below is the code for Out Of Sample R^2 with a print out statement


#OLS_dev <- function(y,pred) {
return( -2*sum((y-pred)^2))
}

#new_y <- test_set_US$SALARY # 

# predictions for 
#OLS_pred <- predict(OLS_Train, newdata=test_set_US) # OLS Prediction
#OLS_pred <- predict(OLS_Train, data=test_set) # logit predictions # I used this commented out code to get a number but I am unsure why my code did not work for the line above
#mean_pred <- mean(training_set_US$SALARY) # predictions of our model


#cat("The Out Of Sample R^2 is ", 1 - OLS_dev(new_y, OLS_pred) / OLS_dev(new_y, mean_pred), ".", sep="")
