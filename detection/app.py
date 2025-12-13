from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)

# Define sequence length
MAX_LENGTH = 100

# Load Tokenizer
tokenizer_path = "models/tokenizer.pkl"
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
else:
    tokenizer = None

# Load RNN Model
model_path = "models/rnn_model.h5"
if os.path.exists(model_path):
    rnn_model = load_model(model_path)
else:
    rnn_model = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/model_eval")
def model_eval():
    return render_template("model_eval.html")

@app.route("/flowchart")
def flowchart():
    return render_template("flowchart.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    global rnn_model, tokenizer
    result = None

    # List of hate speech keywords
    hate_keywords = {"fuck", "rape", "bitch", "racist", "fuck off", "slut", "nigger", "retard", "nigga"}

    if request.method == "POST":
        text = request.form["text"].strip().lower()

        # Split input into words and phrases to match accurately
        words = set(text.split())

        # Keyword-based check (checks both full phrases and individual words)
        if any(keyword in text for keyword in hate_keywords):
            result = "🚨 Hate speech detected (keyword match)"
        else:
            if tokenizer is None:
                return "❌ Tokenizer is missing. Please retrain the model."

            # Prepare text for model prediction
            sequence = tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=MAX_LENGTH, padding="post")

            if rnn_model is None:
                return "❌ Model is missing. Please retrain the model."

            # Predict with model
            pred = rnn_model.predict(padded_sequence)[0][0]
            result = "✅ No hate speech detected" if pred <= 0.5 else "🚨 Hate speech detected"

    return render_template("prediction.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
