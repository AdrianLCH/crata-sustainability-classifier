from flask import Flask, request, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "Crata_model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text_description")

    if text is None:
        return jsonify({"error": "Missing 'text_description' in request body."}), 400

    # Accept either a single string or a list of strings
    if isinstance(text, str):
        texts = [text]
        single = True
    elif isinstance(text, list) and all(isinstance(x, str) for x in text):
        texts = text
        single = False
    else:
        return jsonify({"error": "'text_description' must be a string or a list of strings."}), 400

    preds = model.predict(texts).tolist()

    return jsonify({"prediction": preds[0] if single else preds})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)