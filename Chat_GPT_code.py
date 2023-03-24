#chat gpt

# Import necessary libraries
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load the datasets
Path = os.getcwd()
dataset1 = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Element_Features_Data.xlsx")#reading data set features
dataset2 = pd.read_excel(Path+"\Enhanced-Opposing-Properties-ML\Composition_Properties_Data.xlsx")
 

# Merge the datasets based on a common column
merged_dataset = pd.merge(dataset1, dataset2, on='common_column')

# Split the merged dataset into training and testing sets
X = merged_dataset.drop(['target_variable'], axis=1) # features
y = merged_dataset['target_variable'] # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Create a support vector regression model
model = SVR(kernel='rbf')

# Train the model using the training set
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
print('R-squared:', model.score(X_test, y_test))

# Predict the target variable for a new dataset
new_dataset = pd.read_csv('new_dataset.csv')
new_dataset_scaled = sc_X.transform(new_dataset)
y_new_pred = model.predict(new_dataset_scaled)
