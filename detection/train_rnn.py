import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os

# Define sequence length and vocab size
MAX_LENGTH = 100
VOCAB_SIZE = 10000

# Expanded dummy dataset (you should replace this with real data!)
texts = [
    "I hate you", "You are a bitch", "Kill yourself", "I love this", "Such a nice day",
    "Go fuck yourself", "You are amazing", "He is racist", "You are so beautiful", 
    "That was a lovely speech", "Fuck off", "This is a hate crime", "Have a great day",
    "She is a slut", "You are talented"
]
labels = [
    "hate", "hate", "hate", "normal", "normal", 
    "hate", "normal", "hate", "normal", 
    "normal", "hate", "hate", "normal",
    "hate", "normal"
]

# Tokenization
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding="post")

# Encode labels
label_map = {"hate": 1, "normal": 0}
y = np.array([label_map[label] for label in labels])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Define RNN Model
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LENGTH),
    SimpleRNN(128, return_sequences=False, dropout=0.2),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

# Save the trained model and tokenizer
if not os.path.exists("models"):
    os.makedirs("models")

model.save("models/rnn_model.h5")
with open("models/tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)

print("✅ Model training complete! Model and tokenizer saved in 'models/' folder.")
