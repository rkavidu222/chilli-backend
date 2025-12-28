from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

app = Flask(__name__)

# Load models
cnn_stage1 = load_model("stage1_chili_not_chili.h5")
cnn_stage2 = load_model("stage2_chili_disease.h5")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

def predict_leaf(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Stage 1: chili/not chili
    pred_stage1 = cnn_stage1.predict(arr)[0][0]
    if pred_stage1 >= 0.5:
        return {"result": "❌ Not a Chili Leaf"}

    # Stage 2: disease classification
    preds = cnn_stage2.predict(arr)[0]
    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id] * 100)
    disease_name = class_names[class_id]

    return {"result": f"🌶️ {disease_name} ({confidence:.2f}%)"}

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img = Image.open(file.stream).convert("RGB")
    result = predict_leaf(img)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
