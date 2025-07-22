import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load ML models
ml_models = {
    "Naive Bayes": joblib.load("models/Language_Detection_NB.pkl"),
    "SVM": joblib.load("models/Language_Detection_SVM.pkl"),
    "Logistic Regression": joblib.load("models/Language_Detection_LR.pkl"),
}
vectorizer = joblib.load("models/vectorizer.pkl")

# Load DL components
dl_model = load_model("DL model/language_detection_lstm.h5")
tokenizer = joblib.load("DL model/tokenizer.pkl")
label_encoder = joblib.load("DL model/label_encoder.pkl")

# UI
st.title("🌍 Language Detection App")
st.write("Enter a sentence and select a model to detect the language!")

model_choice = st.selectbox("Select Model", list(ml_models.keys()) + ["Deep Learning (LSTM)"])
user_input = st.text_area("Enter text here:")

if st.button("Detect Language"):
    if user_input.strip() == "":
        st.warning("Please enter some text to detect.")
    else:
        if model_choice == "Deep Learning (LSTM)":
            # Preprocess for DL model
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=50)  # Use same max_len as during training
            pred_idx = np.argmax(dl_model.predict(padded), axis=1)[0]
            prediction = label_encoder.inverse_transform([pred_idx])[0]
        else:
            # Preprocess for ML models
            input_vector = vectorizer.transform([user_input])
            prediction = ml_models[model_choice].predict(input_vector)[0]

        st.success(f"Detected Language using {model_choice}: **{prediction}**")



# python -m streamlit run app.py
