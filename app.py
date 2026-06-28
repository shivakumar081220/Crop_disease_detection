import os
import json
import numpy as np
import time
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# LOAD MODEL
model = load_model("model/plant_model.h5")

# LOAD CLASS LABELS
with open("model/class_indices.json") as f:
    class_indices = json.load(f)

# REVERSE LABELS
labels = {v: k for k, v in class_indices.items()}

IMG_SIZE = 224
UPLOAD_FOLDER = "static"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def load_validation_accuracy():
    csv_path = os.path.join("model", "confusion_matrix.csv")
    if not os.path.isfile(csv_path):
        return None

    try:
        cm = np.loadtxt(csv_path, delimiter=",")
        cm = np.atleast_2d(cm)
        total = cm.sum()
        if total <= 0:
            return None
        return round(float(np.trace(cm) / total) * 100, 2)
    except Exception:
        return None


VAL_ACCURACY = load_validation_accuracy()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def human_label(raw_label):
    return raw_label.replace("_", " ").title()


def confidence_tier(confidence_pct):
    if confidence_pct >= 90:
        return "Very High"
    if confidence_pct >= 75:
        return "High"
    if confidence_pct >= 60:
        return "Moderate"
    return "Low"


def recommendation(confidence_pct):
    if confidence_pct >= 85:
        return "Prediction is stable. You can use this result for first-level field decisions."
    if confidence_pct >= 65:
        return "Prediction is usable, but verify with 1-2 additional leaf photos for safety."
    return "Prediction uncertainty is high. Capture clearer images or validate with an agronomist."


def suggest_treatment(raw_label):
    key = raw_label.lower()

    if "healthy" in key:
        return {
            "title": "Plant looks healthy",
            "steps": [
                "Continue balanced irrigation and avoid overwatering.",
                "Apply preventive nutrition (NPK + micronutrients) as per crop schedule.",
                "Monitor leaves weekly for early symptom changes.",
            ],
            "spray_window": "No curative spray needed. Use preventive bio-fungicide every 10-14 days.",
            "follow_up": "Scan again in 5-7 days or after heavy rain.",
        }

    if any(k in key for k in ["blight", "spot", "scab", "rot", "anthracnose", "smut"]):
        return {
            "title": "Likely fungal infection treatment",
            "steps": [
                "Remove and destroy heavily infected leaves/plant parts.",
                "Spray a recommended fungicide (for example copper-based or mancozeb class) at labeled dose.",
                "Improve airflow and avoid overhead irrigation in evening hours.",
            ],
            "spray_window": "Repeat spray every 7-10 days for 2-3 rounds based on severity.",
            "follow_up": "Rescan after 4-5 days to verify lesion spread reduction.",
        }

    if any(k in key for k in ["rust", "mildew"]):
        return {
            "title": "Rust or mildew management",
            "steps": [
                "Prune infected foliage and keep canopy dry.",
                "Use sulfur/triazole-compatible fungicide as per local extension guidance.",
                "Maintain wider spacing and reduce leaf wetness duration.",
            ],
            "spray_window": "Two sprays at 7-day interval; adjust by weather pressure.",
            "follow_up": "Check new leaves for fresh pustules or powder within 3-4 days.",
        }

    if any(k in key for k in ["virus", "mosaic", "curl", "streak", "tungro"]):
        return {
            "title": "Suspected viral disease response",
            "steps": [
                "Rogue out severely infected plants to reduce source spread.",
                "Control vectors (whitefly/aphid/thrips) using integrated pest management.",
                "Use clean planting material and maintain field sanitation.",
            ],
            "spray_window": "Focus on vector control every 5-7 days during active infestation.",
            "follow_up": "Rescan nearby plants and monitor vector count traps.",
        }

    if any(k in key for k in ["worm", "borer", "beetle", "aphid", "whitefly", "mite", "pests", "miner", "bug"]):
        return {
            "title": "Pest infestation control",
            "steps": [
                "Use yellow/blue sticky traps and remove heavily infested leaves.",
                "Apply bio-control first (neem or Bt where suitable), then selective insecticide if threshold is exceeded.",
                "Rotate chemistry groups to avoid resistance buildup.",
            ],
            "spray_window": "Scout every 3 days; spray only when economic threshold is crossed.",
            "follow_up": "Recheck pest activity after 48-72 hours.",
        }

    if "wilt" in key:
        return {
            "title": "Wilt symptom management",
            "steps": [
                "Improve drainage and avoid waterlogging stress.",
                "Drench root zone with recommended bio-agent or fungicide per diagnosis.",
                "Remove dead plants and sanitize nearby soil/tools.",
            ],
            "spray_window": "Root-zone drench at 7-day interval for 2 cycles.",
            "follow_up": "Observe midday wilting trend for 3-5 days.",
        }

    return {
        "title": "General crop protection plan",
        "steps": [
            "Isolate affected plants and prune damaged tissue.",
            "Use broad integrated management: sanitation, airflow, balanced nutrition.",
            "Consult local agronomy support for crop-specific active ingredient and dose.",
        ],
        "spray_window": "Follow local extension schedule and product label strictly.",
        "follow_up": "Capture a fresh image in 3-5 days for trend comparison.",
    }


def crop_status(predicted_label):
    if "healthy" in predicted_label.lower():
        return "Healthy"
    return "Attention Needed"


def urgency(status_text, confidence_pct):
    if status_text == "Healthy" and confidence_pct >= 80:
        return "Low"
    if confidence_pct >= 70:
        return "Medium"
    return "High"

def predict(img_path):
    start = time.perf_counter()
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    top_index = int(np.argmax(prediction))
    confidence_pct = round(float(prediction[top_index]) * 100, 2)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    top_indices = np.argsort(prediction)[::-1][:3]
    top_predictions = [
        {
            "label": human_label(labels[int(idx)]),
            "confidence": round(float(prediction[int(idx)]) * 100, 2),
        }
        for idx in top_indices
    ]

    predicted_name = human_label(labels[top_index])
    status_text = crop_status(predicted_name)
    urgency_text = urgency(status_text, confidence_pct)
    treatment = suggest_treatment(labels[top_index])

    result = {
        "predicted_class": predicted_name,
        "confidence": confidence_pct,
        "confidence_tier": confidence_tier(confidence_pct),
        "inference_time_ms": elapsed_ms,
        "uncertainty": round(100 - confidence_pct, 2),
        "top_predictions": top_predictions,
        "class_count": len(labels),
        "validation_accuracy": VAL_ACCURACY,
        "recommendation": recommendation(confidence_pct),
        "status": status_text,
        "urgency": urgency_text,
        "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": "plant_model.h5",
        "treatment": treatment,
    }

    return result

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    img_path = None
    uploaded_name = None
    class_names = [human_label(labels[i]) for i in sorted(labels.keys())]

    if request.method == "POST":
        file = request.files.get("file")

        if not file or not file.filename:
            error = "Please choose an image file before running detection."
        elif not allowed_file(file.filename):
            error = "Unsupported file format. Use PNG, JPG, JPEG, BMP, or WEBP."
        else:
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            safe_name = secure_filename(file.filename)
            uploaded_name = safe_name
            unique_name = f"{int(time.time())}_{safe_name}"
            img_path = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(img_path)
            result = predict(img_path)

    return render_template(
        "index.html",
        result=result,
        img_path=img_path,
        error=error,
        uploaded_name=uploaded_name,
        class_names=class_names,
        total_classes=len(class_names),
    )

if __name__ == "__main__":
    app.run(debug=True)