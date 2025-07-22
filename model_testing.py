import joblib

# Load the saved model and vectorizer
model = joblib.load("models/Language_Detection_NB.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

texts = [
    "Hello, how are you?",
    "Hola, ¿cómo estás?",
    "Wie geht es dir?",
    "これは日本語の文章です。",
    "Привет, как дела?"
]

vectors = vectorizer.transform(texts)
predictions = model.predict(vectors)

for text, lang in zip(texts, predictions):
    print(f"{text} → {lang}")