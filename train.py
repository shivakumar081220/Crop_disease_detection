import os
import json
import numpy as np
import tensorflow as tf
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image, ImageFile

from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 🔥 DEVICE CHECK (GPU / DirectML)
# -------------------------------
print("Available Devices:", tf.config.list_physical_devices())

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------
# SETTINGS
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_DIR, "clean_dataset")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

# -------------------------------
# CLEAN BAD IMAGES
# -------------------------------
def quarantine_broken_images(dataset_dir):
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    quarantine_dir = os.path.join(PROJECT_DIR, "bad_images")
    moved = 0

    for root, _, files in os.walk(dataset_dir):
        if os.path.abspath(root).startswith(os.path.abspath(quarantine_dir)):
            continue
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in valid_exts:
                continue

            img_path = os.path.join(root, name)
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.load()
            except Exception:
                rel_path = os.path.relpath(img_path, dataset_dir)
                bad_target = os.path.join(quarantine_dir, rel_path)
                os.makedirs(os.path.dirname(bad_target), exist_ok=True)
                shutil.move(img_path, bad_target)
                moved += 1

    print(f"Moved {moved} broken image(s)" if moved else "No broken images found.")

quarantine_broken_images(DATASET_PATH)

# -------------------------------
# DATA GENERATOR
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # IMPORTANT
)

# -------------------------------
# SAVE CLASS LABELS
# -------------------------------
os.makedirs("model", exist_ok=True)

class_indices = train_data.class_indices
class_names = list(class_indices.keys())

with open("model/class_indices.json", "w") as f:
    json.dump(class_indices, f)

# -------------------------------
# MODEL
# -------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers[-30:]:
    layer.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    train_data.num_classes,
    activation='softmax',
    dtype='float32'
)(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# CALLBACKS
# -------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=2),
    tf.keras.callbacks.ModelCheckpoint(
        "model/best_model.h5",
        save_best_only=True
    )
]

# -------------------------------
# TRAIN
# -------------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# -------------------------------
# 🔥 FIXED EVALUATION
# -------------------------------
y_true = val_data.classes
y_pred_probs = model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_true, y_pred)

# ✅ FIX HERE (IMPORTANT)
labels = np.unique(y_true)

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=[class_names[i] for i in labels]
)

print("\n📊 FINAL METRICS")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)

# -------------------------------
# SAVE MODEL
# -------------------------------
model.save("model/plant_model.h5")

print("✅ Training completed successfully with GPU 🚀")