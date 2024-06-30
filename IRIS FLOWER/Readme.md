# Iris Flower Classification using Support Vector Machine (SVM)

This project demonstrates how to build a machine learning model to classify Iris flowers into three different species using a Support Vector Machine (SVM). The dataset used is the famous Iris dataset. The project includes data visualization, model training, evaluation, and saving/loading the trained model.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Prediction](#prediction)
- [Saving and Loading Model](#saving-and-loading-model)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishal-ravi/cbtcip.git
   cd IRIS FLOWER
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
   ```

## Dataset

The dataset used in this project is stored in an Excel file named `Iris Flower.xlsx`. It contains the following columns:
- Sepal length
- Sepal width
- Petal length
- Petal width
- Class_labels (the species of the Iris flower)

## Features

The features used for classification are:
- Sepal length
- Sepal width
- Petal length
- Petal width

The target variable is the species of the Iris flower.

## Model Training

The Support Vector Machine (SVM) algorithm is used to train the model. The dataset is split into training and testing sets to evaluate the model's performance.

## Evaluation

The model's accuracy is evaluated using the testing set. Additionally, a detailed classification report is generated, showing precision, recall, and F1-score for each class.

## Visualization

The project includes data visualization to explore the dataset and understand the distribution of features. A pair plot is created to visualize the relationships between features, and bar plots are used to show the average values of each feature for all classes.

## Prediction

New samples can be provided to the trained model to predict the species of the Iris flower. The prediction results are printed to the console.

## Saving and Loading Model

The trained SVM model is saved to a file using the `pickle` module. The saved model can be loaded later for making predictions without retraining.

## Usage

To run the project, execute the Python script:

```python
python iris_classification.py
```

This will load the dataset, visualize the data, train the SVM model, evaluate its performance, and print predictions for new samples.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or suggestions, please contact 

---

Happy coding!