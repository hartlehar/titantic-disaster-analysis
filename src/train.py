import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


### Load Data
train = pd.read_csv("data/train.csv")
train = train.dropna()
test = pd.read_csv('data/test.csv')


### Transform Data
# Fill NaN values in Age column of the test set to median age 
# by Pclass and sex from train set
age_medians = train.groupby(['Pclass', 'Sex'])['Age'].median()

def fill_age(row):
    if pd.isna(row['Age']):
        return age_medians.loc[row['Pclass'], row['Sex']]
    else:
        return row['Age']

test['Age'] = test.apply(fill_age, axis=1)

# Convert Sex column into integer, 1 = male and 0 = female
train['Sex'] = (train['Sex'] == 'male').astype(int)
test['Sex'] = (test['Sex'] == 'male').astype(int)

# Convert Cabin into categorical column
train['Cabin'] = train['Cabin'].astype('category')
test['Cabin'] = test['Cabin'].astype('category')

# Convert Embarked into categorical column
train['Embarked'] = train['Embarked'].astype('category')
test['Embarked'] = test['Embarked'].astype('category')


### Train and Evaluate Model
# Initialize model
model = LogisticRegression()
k_folds = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123)

# Select features to be trained on
train_x = train[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass']]
train_y = train['Survived']
test_x = test[['Sex', 'Age', 'SibSp', 'Parch', 'Pclass']]

# Evaluate cross validation performance
scores = cross_val_score(model, train_x, train_y, cv=k_folds, scoring='accuracy')

print("Cross Validation Scores:", scores)
print("Average CV Score:", scores.mean())
print("Standard Deviation of CV Scores:", scores.std())

# Fit on full training data
model.fit(train_x, train_y)
pred = model.predict(train_x)

accuracy_score(train_y, pred)


### Make Predictions
test_pred = model.predict(test_x)

# Save to csv file
results = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})

results.to_csv