# 🌐 Language Detection App

A deep learning and machine learning-based application that detects the language of a given text input. This project supports multiple languages and uses both classical ML models and an LSTM-based neural network for robust detection.

---

## 📁 Project Structure

```
📦Language Detection Model
┣ 📂Dataset
┃ ┗ 📜language.csv
┣ 📂DL model
┃ ┣ 📜language_detection_lstm.h5
┃ ┣ 📜label_encoder.pkl
┃ ┗ 📜tokenizer.pkl
┣ 📂models
┃ ┣ 📜Language_Detection_LR.pkl
┃ ┣ 📜Language_Detection_NB.pkl
┃ ┣ 📜Language_Detection_SVM.pkl
┃ ┣ 📜tokenizer.pkl
┃ ┗ 📜vectorizer.pkl
┣ 📜app.py
┣ 📜DL_model_training.py
┣ 📜model_training.py
┣ 📜model_testing.py
┣ 📜visualizations.py
┣ 📜class labels.txt
```

---

## 🚀 Features

- Detects language from raw text input
- Supports multiple classical ML models: **Logistic Regression**, **Naive Bayes**, **SVM**
- Deep Learning model: **LSTM** trained on tokenized sequences
- Real-time prediction via `app.py`
- Tokenizer, vectorizer, and label encoder saved for inference
- Visualizations for training evaluation

---

## 🧠 Tech Stack

- Python 3.8+
- Scikit-learn
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib
- Git LFS (for large model files)

---

## ⚠️ Git LFS Required

> This repo uses [Git Large File Storage (LFS)](https://git-lfs.github.com) for storing model weights (`.h5`, `.pkl`).

### Install Git LFS before cloning:

```bash
git lfs install
```

Then clone the repo:
```bash
git clone https://github.com/AyushPrasad-ap/Language-Detection-App.git
```

---

## 📊 Training Scripts

DL_model_training.py — trains the LSTM model

model_training.py — trains classical ML models

model_testing.py — evaluation and testing of all models

visualizations.py — generate graphs for model performance

---

## 📬 Contact

Developed by [Ayush Prasad](https://www.linkedin.com/in/ayush-prasad-ap/)😎

Feel free to raise issues or submit pull requests!

---

## 📝 License

This project is licensed under the MIT License.
