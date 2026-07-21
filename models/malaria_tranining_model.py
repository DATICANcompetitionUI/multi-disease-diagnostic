"""
MedAI Nigeria - Malaria Detection Model Training Script
Optimized for local CPU training (no GPU required)
"""

import importlib
from pathlib import Path

try:
    tf = importlib.import_module("tensorflow")
except Exception as e:
    raise ImportError(
        "TensorFlow is not installed or could not be imported.\n"
        "Install it with: pip install tensorflow (or pip install tensorflow-cpu for CPU-only).\n"
        f"Original error: {e}"
    )

MobileNetV2 = tf.keras.applications.MobileNetV2
Dense = tf.keras.layers.Dense
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Dropout = tf.keras.layers.Dropout
Model = tf.keras.models.Model
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Adam = tf.keras.optimizers.Adam

# ============================================
# CONFIG - Adjust these if needed
# ============================================
DATA_DIR = Path("data/cell_images/cell_images")
if not DATA_DIR.exists():
    DATA_DIR = Path("data/cell_images")

IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 3
VALIDATION_SPLIT = 0.2

# ============================================
# STEP 1: Load and preprocess data
# ============================================
print(f"Loading dataset from: {DATA_DIR}")

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=VALIDATION_SPLIT,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

train_generator = datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    str(DATA_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"Class indices: {train_generator.class_indices}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

if len(train_generator.class_indices) != 2:
    raise ValueError(
        f"Expected exactly 2 classes but found {len(train_generator.class_indices)}: "
        f"{train_generator.class_indices}"
    )

# ============================================
# STEP 2: Build the model using MobileNetV2
# ============================================
print("\nBuilding model...")

base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# Use a smaller input size to speed up CPU training.

# Freeze base model layers (faster training, less data needed)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary: Parasitized vs Uninfected

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ============================================
# STEP 3: Train the model
# ============================================
print("\nStarting training...")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ============================================
# STEP 4: Save the model
# ============================================
model.save("malaria_model.keras")
print("\n✅ Model saved as malaria_model.h5")

# ============================================
# STEP 5: Print final results
# ============================================
final_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_acc:.2%}")
print(f"Final Validation Accuracy: {final_val_acc:.2%}")
