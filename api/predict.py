import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from http.server import BaseHTTPRequestHandler
import json

# Load model, scaler, and label encoder
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../rf_star_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../scaler.pkl')
LECOLOR_PATH = os.path.join(os.path.dirname(__file__), '../lecolor.pkl')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
lecolor = joblib.load(LECOLOR_PATH)

# Star type labels
startypelabels = {0:"Brown Dwarf", 1:"Red Dwarf", 2:"White Dwarf", 3:"Main Sequence", 4:"Supergiant", 5:"Hypergiant"}

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data)
            # Expecting: Temperature, Luminosity, Radius, AbsoluteMagnitude, StarColor
            features = [
                data["Temperature"],
                data["Luminosity"],
                data["Radius"],
                data["AbsoluteMagnitude"],
                lecolor.transform([data["StarColor"]])[0]
            ]
            features_scaled = scaler.transform([features])
            pred = model.predict(features_scaled)[0]
            pred_label = startypelabels.get(pred, str(pred))
            response = {"predicted_type": int(pred), "predicted_label": pred_label}
        except Exception as e:
            response = {"error": str(e)}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
