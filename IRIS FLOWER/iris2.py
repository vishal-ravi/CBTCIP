# DataFlair Iris Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the iris dataset from Scikit-learn
iris = datasets.load_iris()
X = iris.data
Y = iris.target
class_labels = iris.target_names

# Convert to DataFrame for visualization purposes
df = pd.DataFrame(X, columns=iris.feature_names)
df['Class_labels'] = [class_labels[i] for i in Y]

# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')
plt.show()

# Calculate average of each feature for all classes
Y_Data = np.array([np.average(X[:, i][Y == j].astype('float32')) for i in range(X.shape[1]) for j in np.unique(Y)])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(iris.feature_names))
width = 0.25

# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa')
plt.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolor')
plt.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica')
plt.xticks(X_axis, iris.feature_names, rotation=45)
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
print(classification_report(y_test, predictions, target_names=class_labels))

# New samples for prediction
X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
# Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format([class_labels[i] for i in prediction]))

# Save the model
with open('SVM.pickle', 'wb') as f:
    pickle.dump(svn, f)

# Load the model
with open('SVM.pickle', 'rb') as f:
    model = pickle.load(f)

# Predict with the loaded model
print("Loaded model predictions:", model.predict(X_new))
