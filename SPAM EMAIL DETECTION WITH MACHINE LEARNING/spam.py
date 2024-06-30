import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Excel file
file_path = 'Spam Email Detection.xlsx'
data = pd.read_excel(file_path)

# Drop unnecessary columns and rename the useful ones
data_cleaned = data[['v1', 'v2']]
data_cleaned.columns = ['label', 'text']

# Convert labels to numerical values
label_encoder = LabelEncoder()
data_cleaned['label'] = label_encoder.fit_transform(data_cleaned['label'])

# Check and clean non-string entries
X = data_cleaned['text']
y = data_cleaned['label']
non_string_mask = X.apply(lambda x: isinstance(x, str))
X_cleaned = X[non_string_mask]
y_cleaned = y[non_string_mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{conf_matrix}')
