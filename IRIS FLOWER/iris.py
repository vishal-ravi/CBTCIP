# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset from the provided Excel file
file_path = 'Iris Flower.xlsx'
iris_data = pd.read_excel(file_path)

# Drop the Id column if present
if 'Id' in iris_data.columns:
    iris_data = iris_data.drop(columns=['Id'])

# Rename columns for consistency
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels']
iris_data.columns = columns

# Features and labels
X = iris_data.drop("Class_labels", axis=1).values
Y = iris_data["Class_labels"].values

# Convert to DataFrame for visualization purposes
df = iris_data

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')
plt.show()

# Calculate average of each feature for all classes
unique_classes = np.unique(Y)
Y_Data = np.array([np.average(X[:, i][Y == j].astype('float32')) for i in range(X.shape[1]) for j in unique_classes])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns) - 1)
width = 0.25

# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label=unique_classes[0])
plt.bar(X_axis + width, Y_Data_reshaped[1], width, label=unique_classes[1])
plt.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label=unique_classes[2])
plt.xticks(X_axis, columns[:4], rotation=45)
plt.xlabel("Features")
plt.ylabel("Value in cm")
plt.legend(bbox_to_anchor=(1.3, 1))
plt.show()

# Split the data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Support vector machine algorithm
svn = SVC()
svn.fit(X_train, y_train)

# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# A detailed classification report
print(classification_report(y_test, predictions))

# New samples for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
# Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))

# Save the model
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# Predict with the loaded model
print("Loaded model predictions:", model.predict(X_new))
