import pandas as pd
import zipfile
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

##### EXTRACT AND READ THE DATA

try:
    train_data = pd.read_csv('C:/Users/Clinton/Desktop/Python/Kaggle Titanic/raw/train.csv')
except:
    with zipfile.ZipFile('C:/Users/Clinton/Desktop/Python/Kaggle Titanic/raw/titanic.zip', 'r') as zip_ref:
        zip_ref.extractall('C:/Users/Clinton/Desktop/Python/Kaggle Titanic/raw')

test_data = pd.read_csv("C:/Users/Clinton/Desktop/Python/Kaggle Titanic/raw/test.csv")

women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women)
# print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
# rate_men = sum(men)/len(men)
# print("% of men who survived:", rate_men)

##### Feature Engineering
### Turning cabin number into Deck
def substrings_in_string(big_string, substrings):
    if isinstance(big_string, str):
        # print(big_string)
        for substring in substrings:
            if big_string.find(substring) != -1:
                return substring
    return np.nan

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'Unknown']
train_data['Deck'] = train_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
test_data['Deck'] = test_data['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))

# Note that a singular cabin on deck "T" exists in the train data. We will remove it as it is a singular data point
# and causes problems with generating the prediction
train_data = train_data[train_data['Deck'] != 'T']

### Creating new family_size column
train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch']
test_data['Family_Size'] = test_data['SibSp'] + test_data['Parch']

### Drop if fare is missing (Only 1 row)
test_data.dropna(subset=['Fare'], inplace=True)

##### Prediction
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Deck", "Family_Size"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
accuracy = np.array(cross_val_score(model, X, y, cv=10, scoring='accuracy'))
print(accuracy, accuracy.mean(), accuracy.std())
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)

"""
### Creating age bands of 10 to handle missing age

# Drop isolated observation of 80-year old.
train_data = train_data[train_data['Age'] != 80]
def band_of_10(number):
    if math.isnan(number):
        return 'Unknown'
    else:
        return str(math.floor(number / 10))
train_data['Age_Band'] = pd.Series(map(band_of_10, pd.Series(train_data['Age'])))
test_data['Age_Band'] = pd.Series(map(band_of_10, pd.Series(test_data['Age'])))
"""
