import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
train_df = pd.read_csv('C:/Users/Naila/Desktop/Corona_NLP_train.csv', encoding='latin-1');
test_df = pd.read_csv('C:/Users/Naila/Desktop/Corona_NLP_test.csv', encoding='latin-1');

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(train_df['OriginalTweet'], train_df['Sentiment'], random_state=0)

# Use CountVectorizer to extract features from the text data
vectorizer = CountVectorizer(stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)

# Apply TfidfTransformer to convert the features to tf-idf measures
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train the logistic regression classifier on the data
clf = LogisticRegression(random_state=0).fit(X_train_tfidf, y_train)

# Test the model and calculate the accuracy
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the confusion matrix to better understand the model's accuracy
confusion_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(confusion_mat)
ax.grid(False)
ax.xaxis.set(ticks=range(len(set(y_test))), ticklabels=set(y_test))
ax.yaxis.set(ticks=range(len(set(y_test))), ticklabels=set(y_test))
for i in range(len(set(y_test))):
    for j in range(len(set(y_test))):
        ax.text(j, i, confusion_mat[i, j], ha='center', va='center', color='white')
plt.show()
