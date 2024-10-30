#jfr
from sklearn.feature_extraction.text import TfidfVectorizer
#Convert raw text into a matrix of TF-IDF features
#Internally, this breaks down tokens into TF-IDF scores. This is used to weigh their relevance to the command
from sklearn.linear_model import LogisticRegression
#Machine learning model
#Models the probability that a command is matched to class (our "answers")
#Uses the sigmoid function to map real-valued number to values between 0 and 1 as the probability.

#Intially, I am defining the data, the labels correspond to each the values of each data token respectively.
data = ["How do I List files", "How do I Copy files", "How do I Move files", "How do I delete files", "how do I search for a string"]
labels = ["ls", "cp", "mv", "rm", "grep"];

#Creates the vectorizer instance, this will convert our data sets into vectors based on the TF-IDF metric.
vectorizer = TfidfVectorizer()
vector_data = vectorizer.fit_transform(data)
#fit_transform: Takes in every piece of "data" (strings in our initial case) and tokenizes them, splitting them down to individual words.
#Calculates the Term Frequency (TF), based on how often the words appear. Then calculates the Inverse Document Frequency (IDF)
#IDF measures the importance of a word based on it's rarity across all of the data. TF * IDF = TF-IDF Matrix
#This outputs a "sparse matrix", where a row represents a document (query) and the columns represent a word in the vocab, with a TF-IDF score.

#Initializes the logistic regression model.
model = LogisticRegression()
model.fit(vector_data, labels)
#fit trains the model with our vectorized data and maps them to the corresponding labels.
#Fit minimizes a loss function that measures how well the model's predictions match the labels.
#The model uses an iterative process (gradient descent) to adjust the parameters to make better predictions.

new_command = ["How do I delete a file"]; #Our prompt

#We use the same vectorizer to convert the text into the same format (TF-IDF Matrix) as the training data.
#The vectorizer uses the same vocublary it learned from the training phase.
command_vectorized = vectorizer.transform(new_command)

#Finally, it predicts the label of the response by using logistic regression to calculate the probability that the input belongs to one of our classes.
#The class with the highest probability is selected, then given as the predicition.
prediction = model.predict(command_vectorized)

print(f"Try using {prediction[0]}")
