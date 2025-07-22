import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
df = pd.read_csv("Dataset/language.csv")
texts = df["Text"].values
labels = df["language"].values

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(np.unique(encoded_labels))
y = to_categorical(encoded_labels, num_classes=num_classes)
print(num_classes)

# Tokenization
max_words = 10000
max_len = 100  # Max length of a sentence (tune this if needed)

tokenizer = Tokenizer(num_words=max_words, char_level=True)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32,activation="relu"),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Training
#early_stop = EarlyStopping(monitor='val_loss' ,restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.15,
    #callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Deep Learning Model Accuracy: {accuracy:.4f}")

# Save model and tokenizer
os.makedirs("DL model", exist_ok=True)
model.save("DL model/language_detection_lstm.h5")
joblib.dump(tokenizer, "DL model/tokenizer.pkl")
joblib.dump(label_encoder, "DL model/label_encoder.pkl")
