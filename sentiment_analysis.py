import pandas as pd
import nltk

df = pd.read_csv("IMDB Dataset.csv")
print(df.head())

import re
import string
nltk.download('stopwords')
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>","",text)
    text = re.sub(r"\d+","",text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(nltk.corpus.stopwords.words("english"))
    words = text.split()
    cleaned = [word for word in words if word not in stop_words]
    text = " ".join(cleaned)
    return text

df["cleaned_review"] = df["review"].apply(clean_text)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df["cleaned_review"])
y = df["sentiment"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state= 42)
print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))



