library(caret)
library(keras)
library(e1071)
library(xgboost)
library(randomForest)
library(gbm)

# Read the genomic data and phenotype data into R data frames
data(wheat)
Y <- wheat.Y
X <- wheat.X
A <- wheat.A

y <- Y[, 1]

# Split the data into training and testing sets
set.seed(123)
train_indices <- createDataPartition(y, p = 0.7, list = FALSE)
train_data <- X[train_indices, ]
test_data <- X[-train_indices, ]
train_phenotype <- y[train_indices]
test_phenotype <- y[-train_indices]

# Define the control settings for model training
control <- trainControl(method = "cv", number = 5)

# Train different machine learning algorithms using cross-validation

# Example 1: Convolutional Neural Network (CNN)
# Update the model_cnn code to use the correct layer structure and parameters
model_cnn <- train(train_data, train_phenotype,
                   method = "knn", trControl = control,
                   tuneGrid = data.frame(k = 3))

# Example 2: K-Nearest Neighbors (KNN)
model_knn <- train(train_data, train_phenotype, method = "knn", trControl = control)

# Example 3: Support Vector Classifier (SVC)
model_svc <- train(train_data, train_phenotype, method = "svmRadial", trControl = control)

# Example 4: XGBoost Classifier (XGB)
model_xgb <- train(train_data, train_phenotype, method = "xgbTree", trControl = control)

# Example 5: Random Forest Classifier (RF)
model_rf <- train(train_data, train_phenotype, method = "rf", trControl = control)

# Example 6: Gradient Boosting Classifier (GB)
model_gb <- train(train_data, train_phenotype, method = "gbm", trControl = control)

# Predict on the testing set using the trained models
pred_cnn <- predict(model_cnn, newdata = test_data)
pred_knn <- predict(model_knn, newdata = test_data)
pred_svc <- predict(model_svc, newdata = test_data)
pred_xgb <- predict(model_xgb, newdata = test_data)
pred_rf <- predict(model_rf, newdata = test_data)
pred_gb <- predict(model_gb, newdata = test_data)

# Evaluate model performance
accuracy_cnn <- sum(pred_cnn == test_phenotype) / length(test_phenotype)
accuracy_knn <- sum(pred_knn == test_phenotype) / length(test_phenotype)
accuracy_svc <- sum(pred_svc == test_phenotype) / length(test_phenotype)
accuracy_xgb <- sum(pred_xgb == test_phenotype) / length(test_phenotype)
accuracy_rf <- sum(pred_rf == test_phenotype) / length(test_phenotype)
accuracy_gb <- sum(pred_gb == test_phenotype) / length(test_phenotype)

# Print the accuracy for each model
cat("Accuracy (CNN):", accuracy_cnn, "\n")
cat("Accuracy (KNN):", accuracy_knn, "\n")
cat("Accuracy (SVC):", accuracy_svc, "\n")
cat("Accuracy (XGB):", accuracy_xgb, "\n")
cat("Accuracy (RF):", accuracy_rf, "\n")
cat("Accuracy (GB):", accuracy_gb, "\n")
