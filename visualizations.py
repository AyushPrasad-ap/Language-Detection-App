import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Load saved models and vectorizer
nb_model = joblib.load("models/Language_Detection_NB.pkl")
svm_model = joblib.load("models/Language_Detection_SVM.pkl")
lr_model = joblib.load("models/Language_Detection_LR.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Load the dataset
data = pd.read_csv("Dataset/language.csv")
x = np.array(data["Text"])
y = np.array(data["language"])

# Preprocess
X = vectorizer.transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Accuracy Scores
accuracies = {
    "Naive Bayes": nb_model.score(X_test, y_test),
    "SVM": svm_model.score(X_test, y_test),
    "Logistic Regression": lr_model.score(X_test, y_test)
}

# Plot
colors = ["#FFADAD", "#FFD6A5", "#CAFFBF"]

plt.figure(figsize=(9, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)

# Add accuracy labels on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.2%}", ha='center', va='bottom', fontsize=12)

# Styling
plt.title("Model Accuracy Comparison", fontsize=16)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0.8, 1.0)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()






# ------------------------------------------ Pie Chart ------------------------------------------

import matplotlib.pyplot as plt

# Count each language
lang_counts = data['language'].value_counts()

# Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
plt.title("Language Distribution in Dataset", fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()





# ------------------------- Bar Chart: Sample Text Length Distribution ---------------------------


data['text_length'] = data['Text'].apply(len)

plt.figure(figsize=(10, 5))
plt.hist(data['text_length'], bins=30, color="#A0C4FF", edgecolor='black')
plt.title("Distribution of Text Lengths", fontsize=14)
plt.xlabel("Number of Characters")
plt.ylabel("Number of Samples")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()





# ------------------------- Top N Most Frequent Characters (in all text) -------------------------

from collections import Counter

all_text = " ".join(data["Text"].values)
char_counts = Counter(all_text)

# Get top 10
top_chars = dict(char_counts.most_common(10))

plt.figure(figsize=(8, 5))
plt.bar(top_chars.keys(), top_chars.values(), color="#BDB2FF")
plt.title("Top 10 Most Common Characters", fontsize=14)
plt.xlabel("Character")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# ------------------------- Word Cloud of Most Frequent Words -------------------------

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all text
all_text = " ".join(data["Text"].astype(str))

# Generate Word Cloud
wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    max_words=200,
    colormap='viridis'
).generate(all_text)

# Plot
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of All Texts", fontsize=16)
plt.tight_layout()
plt.show()
