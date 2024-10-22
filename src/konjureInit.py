import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = []
vectorizer = CountVectorizer()
transformed_data = vectorizer.fit_transform(data)
