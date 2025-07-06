# WomensClothing_NLP_Capstone.ipynb (Python script version)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
import joblib
import os

# Load and preprocess data
file_path = "Womens Clothing Reviews Data.xlsx"
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
df.dropna(subset=['rating', 'customer_age', 'review_text', 'recommend_flag'], inplace=True)

# Add sentiment score
df['sentiment'] = df['review_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Age group bins
bins = [18, 25, 35, 45, 55, 65, 100]
labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
df['age_group'] = pd.cut(df['customer_age'], bins=bins, labels=labels, right=False)

# EDA Plots
sns.countplot(x='rating', data=df)
plt.title("Rating Distribution")
plt.savefig("rating_distribution.png")
plt.close()

wc = WordCloud(background_color='white', max_words=100).generate(" ".join(df['review_text']))
wc.to_file("wordcloud_all_reviews.png")

# TF-IDF Features
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['review_text'])

# Classification: Recommend Flag
X_cls = X_text
y_cls = df['recommend_flag']
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

models_cls = {
    'LogReg': LogisticRegression(max_iter=200),
    'NaiveBayes': MultinomialNB(),
    'RandomForest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'SVM': SVC()
}

for name, model in models_cls.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"\nModel: {name}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    joblib.dump(model, f"model_recommend_{name}.joblib")

# Regression: Predict Rating
y_reg = df['rating']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_text, y_reg, test_size=0.2, random_state=42)

models_reg = {
    'Linear': LinearRegression(),
    'RFRegressor': RandomForestRegressor(),
    'XGBRegressor': XGBRegressor()
}

for name, model in models_reg.items():
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)
    mse = mean_squared_error(y_test_r, preds)
    print(f"\n{name} MSE: {mse:.2f}")
    joblib.dump(model, f"model_rating_{name}.joblib")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
