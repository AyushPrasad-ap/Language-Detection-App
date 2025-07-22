import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("Dataset/language.csv")

x = np.array(data["Text"])
y = np.array(data["language"])

# vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X = vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Naive Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Support Vector Classifier
svm_model = SVC(kernel='linear')  
svm_model.fit(X_train, y_train)

# Logistic Regression Classifier
lr_model = LogisticRegression(max_iter=1000)  
lr_model.fit(X_train, y_train)


os.makedirs("models", exist_ok=True)
joblib.dump(nb_model, "models/Language_Detection_NB.pkl")
joblib.dump(svm_model, "models/Language_Detection_SVM.pkl")
joblib.dump(lr_model, "models/Language_Detection_LR.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")


print("\nTraining complete. Models saved successfully!\n")
print("Naive Bayes - Accuracy:", nb_model.score(X_test, y_test))
print("SVM - Accuracy:", svm_model.score(X_test, y_test))
print("Logistic Regression - Accuracy:", lr_model.score(X_test, y_test))


