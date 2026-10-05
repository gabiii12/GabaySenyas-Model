"""
Train an LSTM on recorded sign clips.

Usage:
    python train_sequences.py

Reads:   data/sequences/<sign>/<signer>_<n>.npy
Writes:  model/fsl_sequence_model.keras
         model/sequence_labels.json

If there are 3+ different signers, whole signers are held out for testing
(the honest test). Otherwise it falls back to a random split and warns you.
"""
import glob
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

ROOT = "data/sequences"
os.makedirs("model", exist_ok=True)

signs = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
X, y, groups = [], [], []
for i, sign in enumerate(signs):
    for path in glob.glob(os.path.join(ROOT, sign, "*.npy")):
        X.append(np.load(path))
        y.append(i)
        groups.append(os.path.basename(path).rsplit("_", 1)[0])  # signer id
X, y, groups = np.array(X, dtype="float32"), np.array(y), np.array(groups)

print(f"{len(X)} clips, {len(signs)} signs: {signs}")
print(f"Signers: {sorted(set(groups))}")
if len(signs) < 2:
    raise SystemExit("Need at least 2 different signs to train.")

with open("model/sequence_labels.json", "w") as f:
    json.dump(signs, f)

if len(set(groups)) >= 3:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    print("Split by SIGNER (honest test).")
else:
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
    print("WARNING: fewer than 3 signers, random split. Score will be optimistic.")

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X.shape[1:]),
    tf.keras.layers.GaussianNoise(0.02),  # light augmentation (training only)
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(signs), activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(
    X_train, y_train, validation_split=0.15, epochs=150, batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)],
    verbose=2)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc:.3f}")
pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
print(classification_report(y_test, pred, labels=range(len(signs)),
                            target_names=signs, zero_division=0))

model.save("model/fsl_sequence_model.keras")
print("Saved model/fsl_sequence_model.keras and model/sequence_labels.json")