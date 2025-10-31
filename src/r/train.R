library(dplyr)

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
cat("Dropped 'Cabin' column since it has many missing values.\n")

# Convert Embarked to categorical
train$Embarked <- as.factor(train$Embarked)
test$Embarked <- as.factor(test$Embarked)
cat("Converted 'Embarked' column to categorical variables.\n")


# ---- Train and Evaluate Model ----
cat("\n\n---- Training Model ----\n")

# Select features
features <- c("Sex", "Age", "SibSp", "Parch", "Pclass")
train_x <- train[, features]
train_y <- train$Survived

# Combine for glm
train_df <- cbind(train_x, Survived = train_y)

# Fit logistic regression
model <- glm(Survived ~ ., data = train_df, family = binomial)

cat("Model trained using glm with predictors: ", paste(features, collapse = ", "), "\n")


# ---- Evaluate on Training Set ----
train_pred_probs <- predict(model, newdata = train_x, type = "response")
train_pred <- ifelse(train_pred_probs > 0.5, 1, 0)
train_acc <- mean(train_pred == train_y)
cat(sprintf("Training Accuracy: %.4f\n", train_acc))


# ---- Make Predictions on Test Set ----
test_x <- test[, features]
test_pred_probs <- predict(model, newdata = test_x, type = "response")
test_pred <- ifelse(test_pred_probs > 0.5, 1, 0)

# Save predictions
results <- data.frame(
  PassengerId = test$PassengerId,
  Survived = test_pred
)
cat(nrow(results), "predictions made.\n")

cat("Preview of predictions:\n")
print(head(results))

write.csv(results, "test_predictions.csv", row.names = FALSE)
cat("Predictions saved to test_predictions.csv\n")

