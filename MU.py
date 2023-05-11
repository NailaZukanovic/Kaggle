import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#Zatim ćemo učitati skup podataka koji se sastoji od tekstualnih podataka i odgovarajućih klasifikacija.

train_df = pd.read_csv('C:/Users/Naila/Desktop/Corona_NLP_train.csv', encoding='latin-1');
test_df = pd.read_csv('C:/Users/Naila/Desktop/Corona_NLP_test.csv', encoding='latin-1');

#Kao što je ovaj skup podataka već razdvojen na testni i obučavajući skup podataka, podatke ćemo podijeliti u obučavajući i testni skup podataka.

X_train, X_test, y_train, y_test = train_test_split(train_df['OriginalTweet'], train_df['Sentiment'], random_state=0)

#Sada ćemo koristiti CountVectorizer za izdvajanje značajki iz tekstualnih podataka.

vectorizer = CountVectorizer(stop_words='english')
X_train_counts = vectorizer.fit_transform(X_train)

#Nakon toga ćemo primijeniti TfidfTransformer kako bismo pretvorili značajke u Tfidf mjere.

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#Sada smo spremni da primijenimo klasifikator i treniramo ga na našim podacima.

clf = MultinomialNB().fit(X_train_tfidf, y_train)

#Nakon što smo obučili klasifikator, sada ćemo testirati naše modele i izračunati tačnost.

X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

#Sada ćemo vizualizovati konfuzijsku matricu kako bismo bolje razumjeli tačnost našeg modela.

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
