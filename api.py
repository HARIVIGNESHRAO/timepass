# api.py

import argparse
import io
import json
from flask import Flask, request
from PIL import Image
import torch

app = Flask(__name__)

# Load stored API keys
try:
    with open("keys.json", "r") as f:
        API_KEYS = set(json.load(f))
except:
    API_KEYS = set()

models = {}
DETECTION_URL = "/v1/object-detection/<model>"


# -----------------------------
#  API ENDPOINT FOR YOLO
# -----------------------------
@app.route(DETECTION_URL, methods=["POST"])
def predict(model):

    # 1. Check API key
    key = request.headers.get("x-api-key")
    if key not in API_KEYS:
        return {"error": "Invalid or missing API key"}, 401

    # 2. Check image
    if "image" not in request.files:
        return {"error": "No image provided"}, 400

    # 3. Read image
    im_file = request.files["image"]
    im_bytes = im_file.read()
    im = Image.open(io.BytesIO(im_bytes))

    # 4. Check model exists
    if model not in models:
        return {"error": f"Model '{model}' not loaded"}, 404

    # 5. Run YOLO
    results = models[model](im, size=640)
    return results.pandas().xyxy[0].to_json(orient="records")


# -----------------------------
#  API KEY REGISTRATION
# -----------------------------
@app.route("/register-key", methods=["POST"])
def register_key():
    data = request.json
    new_key = data.get("api_key")

    if not new_key:
        return {"error": "Key missing"}, 400

    API_KEYS.add(new_key)

    # Save back to file
    with open("keys.json", "w") as f:
        json.dump(list(API_KEYS), f)

    return {"message": "API key registered", "api_key": new_key}


# -----------------------------
#  MAIN
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int)
    parser.add_argument("--model", nargs="+", default=["yolov5s"])
    opt = parser.parse_args()

    # Load models
    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True)

    app.run(host="0.0.0.0", port=opt.port)
