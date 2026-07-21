import json
import joblib
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("breast_cancer_data/data.csv")

# Remove unnecessary columns
df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

# Convert diagnosis labels
# M = Malignant (Cancer)
# B = Benign (No Cancer)
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# ---------------------------
# Features and Labels
# ---------------------------
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ---------------------------
# Split Dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------
# Normalize Features
# ---------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# Build Neural Network
# ---------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30,)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# ---------------------------
# Compile Model
# ---------------------------
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# Train Model
# ---------------------------
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    verbose=1
)

# ---------------------------
# Evaluate Model
# ---------------------------
predictions = (model.predict(X_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)

print(f"\nTest Accuracy: {accuracy:.4f}")

# ---------------------------
# Save Trained Model
# ---------------------------
model.save("breast_cancer_model.keras")

print("✅ Model saved as breast_cancer_model.keras")

# ---------------------------
# Save StandardScaler
# ---------------------------
joblib.dump(scaler, "breast_cancer_scaler.pkl")

print("✅ Scaler saved as breast_cancer_scaler.pkl")

# ---------------------------
# Save Feature Names
# ---------------------------
feature_names = list(X.columns)

with open("feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=4)

print("✅ Feature names saved as feature_names.json")

# ---------------------------
# Save Class Labels
# ---------------------------
class_labels = [
    "Benign",
    "Malignant"
]

with open("class_labels.json", "w") as f:
    json.dump(class_labels, f, indent=4)

print("✅ Class labels saved as class_labels.json")

print("\n🎉 Training completed successfully!")