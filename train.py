"""
Train a sign classifier on hand-landmark data.

Usage:
    python train.py                      # uses data/landmarks_dataset.csv
    python train.py data/landmarks.csv   # or another CSV

Outputs:
    model/fsl_model.keras   trained model
    model/labels.json       class order (needed by the web app!)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CSV = sys.argv[1] if len(sys.argv) > 1 else "data/landmarks_dataset.csv"
os.makedirs("model", exist_ok=True)

df = pd.read_csv(CSV)
feature_cols = [c for c in df.columns if c.startswith("p")]
X = df[feature_cols].values.astype("float32")

labels = sorted(df["label"].astype(str).unique())
label_to_id = {l: i for i, l in enumerate(labels)}
y = df["label"].astype(str).map(label_to_id).values

with open("model/labels.json", "w") as f:
    json.dump(labels, f)
print(f"{len(df)} samples, {len(labels)} classes: {labels}")

# 70% train / 15% val / 15% test
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(len(labels), activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("model/fsl_model.keras", save_best_only=True),
]

model.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=200, batch_size=32, callbacks=callbacks, verbose=2)

# Evaluate on held-out test data
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.3f}")

pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, pred, target_names=labels, zero_division=0))

# Most common mix-ups
cm = confusion_matrix(y_test, pred)
np.fill_diagonal(cm, 0)
pairs = [(cm[i, j], labels[i], labels[j])
         for i in range(len(labels)) for j in range(len(labels)) if cm[i, j] > 0]
pairs.sort(reverse=True)
print("Most confused (count, true -> predicted):")
for n, t, p in pairs[:10]:
    print(f"  {n}x  {t} -> {p}")

model.save("model/fsl_model.keras")
print("\nSaved model/fsl_model.keras and model/labels.json")