library(dplyr)
library(readr)
library(caret)

set.seed(123)

# ---- Load Data ----
train <- read.csv("data/train.csv")
train <- na.omit(train)
test <- read.csv("data/test.csv")

cat("---- Data Overview ----\n")
cat("Train Set:\n")
str(train)
cat("\nTest Set:\n")
str(test)


# ---- Transform Data ----
cat("\n\n---- Transforming Data ----\n")

# Fill missing Age values in test set with median Age by Pclass and Sex from train
age_medians <- train %>%
  group_by(Pclass, Sex) %>%
  summarize(median_age = median(Age, na.rm = TRUE))

test <- test %>%
  left_join(age_medians, by = c("Pclass", "Sex")) %>%
  mutate(Age = ifelse(is.na(Age), median_age, Age)) %>%
  select(-median_age)

cat("Filled missing Age values in test set using median Age by Pclass and Sex.\n")

# Convert Sex to numeric (1 = male, 0 = female)
train$Sex <- ifelse(train$Sex == "male", 1, 0)
test$Sex <- ifelse(test$Sex == "male", 1, 0)
cat("Converted 'Sex' column to binary variable (1 = male, 0 = female).\n")

# Drop Cabin Column
train <- train %>% select(-Cabin)
test <- test  %>% select(-Cabin)
cat("Dropped 'Cabin' column since it has many missing values.")

# Convert Embarked to categorical
train$Embarked <- as.factor(train$Embarked)
test$Embarked <- as.factor(test$Embarked)
cat("Converted 'Embarked' column to categorical variables.\n")


# ---- Train and Evaluate Model ----
cat("\n\n---- Training and Evaluation ----\n")

cat("Since there is only 183 recordings in the training set, K-folds (specifically 5 folds) validation \
will be used to measure the model's performance.\n")

# Select features and target
features <- c("Sex", "Age", "SibSp", "Parch", "Pclass")
train_x <- train[, features]
train_y <- train$Survived
test_x <- test[, features]

cat("Using features:", paste(features, collapse = ", "), "\n")

# Create 5-fold cross-validation setup
train_control <- trainControl(method = "cv", number = 5)

# Train logistic regression model using caret
model <- train(
  x = train_x,
  y = as.factor(train_y),
  method = "glm",
  family = "binomial",
  trControl = train_control,
  metric = "Accuracy"
)

cat("Cross-validation accuracy results:\n")
print(model$resample$Accuracy)
cat(sprintf("Average CV Score: %.4f\n", mean(model$resample$Accuracy)))
cat(sprintf("Standard Deviation of CV Scores: %.4f\n", sd(model$resample$Accuracy)))


# Evaluate on full training set
train_pred <- predict(model, newdata = train_x)
train_acc <- mean(train_pred == as.factor(train_y))
cat(sprintf("Model Accuracy on Entire Training Set: %.4f\n", train_acc))


# ---- Make Predictions ----
test_pred <- predict(model, newdata = test_x)

# Save to CSV
results <- data.frame(
  PassengerId = test$PassengerId,
  Survived = as.integer(as.character(test_pred))
)

write.csv(results, "test_predictions.csv", row.names = FALSE)
cat("\nTest set predictions saved in 'test_predictions.csv' (PassengerId, Survived)\n")
