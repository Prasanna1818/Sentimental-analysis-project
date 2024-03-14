#Task 3
#Sentiment analysis
import pandas as pd
dataset_path = 'Twitter_Data.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())
dataset_path = 'Twitter_Data.csv'
df = pd.read_csv(dataset_path)

# Display the first few rows of the dataset
print(df.head())

# Check the shape of the dataset
print("Dataset shape:", df.shape)

# Check for any missing values
print("Missing values:\n", df.isnull().sum())

# Check the distribution of sentiment labels 
print("Sentiment distribution:\n", df['category'].value_counts())

#Preprocess the Text Data
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    try:
        if pd.isna(text):
            return ''
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word not in string.punctuation]
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    except Exception as e:
        print("Error:", e)
        return ''

# Apply preprocessing to text data
df['clean_text'] = df['clean_text'].apply(preprocess_text)

#Split the Dataset
from sklearn.model_selection import train_test_split


# Handle missing values in the 'category' column
mode_category = df['category'].mode()[0]
df['category'].fillna(mode_category, inplace=True)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['category'], test_size=0.2, random_state=42)

#Train and Evaluate the Model (Using Naive Bayes Classifier)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Vectorizing text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Making predictions
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.model_selection import GridSearchCV
# Define hyperparameters to tune
param_grid = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False]
}
grid_search = GridSearchCV(nb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

#Cross-Validation
from sklearn.model_selection import cross_val_score
import numpy as np
cv_scores = cross_val_score(nb_classifier, X_train_tfidf, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

def test_model_accuracy():
    assert accuracy >= 0.7, "Model accuracy is too low"


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('Twitter_Data.csv')  # Update the file path with your dataset

# Assuming your dataset has 'sentiment' column containing sentiment labels (positive, negative, neutral)
sentiment_counts = data['category'].value_counts()

# Bar plot for sentiment distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of Categories')
plt.xlabel('category')
plt.ylabel('Count')
plt.show()

# Pie chart for sentiment distribution
plt.figure(figsize=(8, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen'])
plt.title('Distribution of Categories')
plt.show()
