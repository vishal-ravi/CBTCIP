# Spam Email Detection using Naive Bayes

This project demonstrates how to build a machine learning model to detect spam emails using a Naive Bayes classifier. The dataset used is a collection of emails labeled as spam or ham (non-spam). The project includes data cleaning, feature extraction, model training, evaluation, and visualization.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishal-ravi/cbtcip.git
   cd spam-email-detection
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn openpyxl
   ```

## Dataset

The dataset used in this project is stored in an Excel file named `Spam Email Detection.xlsx`. It contains the following columns:
- `v1`: Label (either "ham" or "spam")
- `v2`: Email text

## Features

The features used for classification are extracted from the email text using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This converts the text data into numerical features that can be used by the machine learning model.

## Model Training

The Naive Bayes classifier is used to train the model. The dataset is split into training and testing sets to evaluate the model's performance. The `MultinomialNB` classifier from `scikit-learn` is used.

## Evaluation

The model's accuracy is evaluated using the testing set. Additionally, a detailed classification report is generated, showing precision, recall, and F1-score for each class. A confusion matrix is also created to visualize the model's performance in distinguishing between spam and ham emails.

## Visualization

The project includes visualizations to understand the model's performance better:
- **Confusion Matrix**: Shows the number of true positive, true negative, false positive, and false negative predictions.
- **ROC Curve**: Illustrates the model's ability to distinguish between classes at various threshold settings, along with the Area Under the Curve (AUC) metric.

## Usage

To run the project, execute the Python script:

```python
python spam.py
```

This will load the dataset, clean the data, convert text to TF-IDF features, train the Naive Bayes model, evaluate its performance, and generate visualizations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or suggestions

---

Happy coding!