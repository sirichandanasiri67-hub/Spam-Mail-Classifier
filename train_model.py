import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

# Sample dataset (you can replace later)
data = {
    "text": [
        "Win a free lottery now",
        "Claim your prize click here",
        "Meeting at 10 AM tomorrow",
        "Project submission deadline",
        "Free offer limited time",
        "Important account verification required"
    ],
    "label": [
        "Lottery Scam",
        "Phishing Mail",
        "Normal Mail",
        "Normal Mail",
        "Advertisement Spam",
        "Phishing Mail"
    ]
}

df = pd.DataFrame(data)

# Convert text → numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")
