# 🎬 IMDb Sentiment Analysis using Python & Naive Bayes

This project is a simple **Natural Language Processing (NLP)** pipeline that classifies IMDb movie reviews as **positive** or **negative** using Python and machine learning.

---

## 📌 **What this project does**

- Loads IMDb review data (50,000 movie reviews)
- Cleans the text:
  - Lowercases all text
  - Removes HTML tags, punctuation, and numbers
  - Removes stopwords (like "the", "and", "is", etc.)
- Transforms the reviews into numeric features using `CountVectorizer`
- Trains a **Naive Bayes** classifier (`MultinomialNB`)
- Predicts the sentiment of unseen reviews
- Evaluates the model using accuracy, precision, recall, and F1-score

---

## 🛠️ **Tools & Libraries used**

- Python 🐍
- `pandas` for data handling
- `nltk` for text processing
- `scikit-learn` for machine learning

---

## 📊 **Example Output**

          precision    recall  f1-score   support

negative       0.88      0.87      0.88      5000
positive       0.88      0.89      0.88      5000

accuracy                           0.88     10000

---

## 🧪 **Sample Prediction**

```python
sample = ["This movie was boring and way too long."]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)
print("Prediction:", prediction[0])
```

---

### 📁 **Dataset**

This project uses the publicly available [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

### 📌 **How this project was made**

This project was built as part of my self-learning in Natural Language Processing.  
It was developed step-by-step with explanations using guidance from OpenAI's ChatGPT.

---

### ✨ **Author**

**Nasim Sadeghi**




---

