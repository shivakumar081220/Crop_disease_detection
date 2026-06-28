import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFile
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
BATCH_SIZE = 32

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_DIR, "clean_dataset")
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "plant_model.h5")
CLASS_INDEX_PATH = os.path.join(PROJECT_DIR, "model", "class_indices.json")
OUTPUT_IMAGE_PATH = os.path.join(PROJECT_DIR, "model", "confusion_matrix.png")
OUTPUT_CSV_PATH = os.path.join(PROJECT_DIR, "model", "confusion_matrix.csv")

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_class_names(index_json_path):
    with open(index_json_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)

    # Reverse mapping from index -> class name.
    idx_to_class = {idx: name for name, idx in class_indices.items()}
    return idx_to_class


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


def main():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    if not os.path.isfile(CLASS_INDEX_PATH):
        raise FileNotFoundError(f"class_indices.json not found: {CLASS_INDEX_PATH}")

    quarantine_broken_images(DATASET_PATH)

    idx_to_class = load_class_names(CLASS_INDEX_PATH)

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )

    val_data = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = val_data.classes
    y_pred_probs = model.predict(val_data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    display_labels = [idx_to_class[i] for i in labels]

    os.makedirs(os.path.join(PROJECT_DIR, "model"), exist_ok=True)

    np.savetxt(OUTPUT_CSV_PATH, cm, fmt="%d", delimiter=",")

    fig_w = max(12, int(len(display_labels) * 0.35))
    fig_h = max(10, int(len(display_labels) * 0.35))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        include_values=False,
        colorbar=True,
    )

    ax.set_title("Validation Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMAGE_PATH, dpi=300)

    print(f"Saved confusion matrix image: {OUTPUT_IMAGE_PATH}")
    print(f"Saved confusion matrix csv: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
