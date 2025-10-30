import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


### Load Data
train = pd.read_csv("data/train.csv")
train = train.dropna()
test = pd.read_csv('data/test.csv')

print("---- Data Overview ----")
print("Train Set:")
print(train.info())
print()
print("Test Set:")
print(test.info())


### Transform Data
print("\n\n---- Transforming Data ----")

# Fill NaN values in Age column of the test set to median age 
# by Pclass and sex from train set
age_medians = train.groupby(['Pclass', 'Sex'])['Age'].median()

def fill_age(row):
    if pd.isna(row['Age']):
        return age_medians.loc[row['Pclass'], row['Sex']]
    else:
        return row['Age']

test['Age'] = test.apply(fill_age, axis=1)
print("The `test` set has several NaN values in the `Age` columns. \
These NaN values will be replaced with the median age based on the passenger's age and their ticket class.")

# Convert Sex column into integer, 1 = male and 0 = female
train['Sex'] = (train['Sex'] == 'male').astype(int)
test['Sex'] = (test['Sex'] == 'male').astype(int)
print("The `Sex` column is a string which cannot be used in logistic regression. \
Converting it into a binary variable makes it compatible.")

# Convert Cabin into categorical column
train['Cabin'] = train['Cabin'].astype('category')
test['Cabin'] = test['Cabin'].astype('category')
print("The `Cabin` column is a string which cannot be used in logistic regression. \
Converting it into a categorical variable.")

# Convert Embarked into categorical column
train['Embarked'] = train['Embarked'].astype('category')
test['Embarked'] = test['Embarked'].astype('category')
print("The `Embarked` column is a string which cannot be used in logistic regression. \
Converting it into a categorical variable.")

### Train and Evaluate Model
print("\n\n---- Training and Evaluation ----")

# Initialize model
model = LogisticRegression()
k_folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123)
print("Since there is only 183 recordings in the training set, K-folds (specifically 5 folds) validation \
will be used to measure the model's performance.")

# Select features to be trained on
train_x = train[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass']]
train_y = train['Survived']
test_x = test[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass']]
print("'Sex', 'Age', 'SibSp', 'Parch', 'Pclass' will be the features used in the model.")

# Evaluate cross validation performance
scores = cross_val_score(model, train_x, train_y, cv=k_folds, scoring='accuracy')

print("Cross Validation Accuracy Scores:", scores)
print(f"Average CV Score: {scores.mean():.4f}")
print(f"Standard Deviation of CV Scores: {scores.std():.4f}")

# Fit on full training data
model.fit(train_x, train_y)
pred = model.predict(train_x)

print(f"Model Accuracy on Entire Training Set: {accuracy_score(train_y, pred):.4f}")


### Make Predictions
test_pred = model.predict(test_x)

# Save to csv file
results = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})

results.to_csv("test_predictions.csv", index=False)
print("\nTest set predictions saved in 'test_predictions.csv' with 2 columns: PassengerId, Survived")