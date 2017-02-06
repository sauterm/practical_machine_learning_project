
library(caret)
library(rattle)

training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

# The outcome we're interested in
plot(training$classe)

# There are a lot of missing values in the data. Quite a few of the columns 
# have mostly NA, and others are blank. Let's see:
# For each column sum up the number of NA or empty values
missing_data_sums <- colSums(is.na(training) | training == "")

# From the histogram we can see that the columns break into one 
# of two categories - ones with very few NA values, and ones with 
# a lot of missing values.
hist(missing_data_sums)

# There are 100 of these
empty_cols <- names(missing_data_sums[missing_data_sums > 19000])

# Let's remove them from the original data set
training_2 <- training[, !(colnames(training) %in% empty_cols)]

# This leaves us with only 59 (60 - 1) predictors to worry about.
dim(training_2)

# Take out the "X" variables - it's a row counter of some sort
training_2 <- training_2[ , -which(names(training_2) %in% c("X"))]

# Let's shuffle training_2
training_2 <- training_2[sample(nrow(training_2 )), ]

# Subset of this for actual training 
training_3 <- training_2[1:2000,]
# Take the next 1000 entries for cross-validation
cross_val <- training_2[2001:3000,]

# LDA
Sys.time()
modelLDA <- train(classe ~ ., method="lda", data=training_3)
Sys.time()
predsLDA <- predict(modelLDA, newdata = cross_val)
crosstabLDA <- table(predsLDA, cross_val$classe)
(sum(diag(crosstabLDA)) * 100)/1000
testPredictionsLDA <- predict(modelLDA, newdata = testing)

# Classification tree
Sys.time()
modelRpart <- train(classe ~ ., method="rpart", data=training_3)
Sys.time()
#fancyRpartPlot(modelRpart$finalModel)
predsRpart <- predict(modelRpart, newdata = cross_val)
crosstabRpart <- table(predsRpart, cross_val$classe)
(sum(diag(crosstabRpart)) * 100)/1000
testPredictionsRPart <- predict(modelRpart, newdata = testing)

# Random Forest 
Sys.time()
modelRF <- train(classe ~ ., method="rf", data=training_3)
Sys.time()
#varImpPlot(modelRF$finalModel)
predsRF <- predict(modelRF, newdata = cross_val)
crosstabRF <- table(predsRF, cross_val$classe)
(sum(diag(crosstabRF)) * 100)/1000
testPredictionsRF <- predict(modelRF, newdata = testing)

# GBM - Generalised Boosted Models
Sys.time()
modelGBM <- train(classe ~ ., method="gbm", data=training_3, verbose=FALSE)
Sys.time()
predsGBM <- predict(modelGBM, newdata = cross_val)
crosstabGBM <- table(predsGBM, cross_val$classe)
(sum(diag(crosstabGBM)) * 100)/1000
testPredictionsGBM <- predict(modelGBM, newdata = testing)


testPredictionsLDA 
testPredictionsRPart 
testPredictionsRF 
testPredictionsGBM 

