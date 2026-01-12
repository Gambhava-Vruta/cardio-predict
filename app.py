# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# from flask import send_from_directory
# import os
# app = Flask(__name__)
# CORS(app)

# THRESHOLD = 0.4 

# # Load RandomForest model
# print("📦 Loading RandomForest model...")
# try:
#     model = joblib.load('rf_model.pkl')
#     print("✅ RandomForest model loaded!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# # Load scaler (if used in training)
# try:
#     scaler = joblib.load('scaler.pkl')
#     USE_SCALER = hasattr(scaler, 'transform')
#     print(f"✅ Scaler loaded: {type(scaler)}")
# except Exception:
#     scaler = None
#     USE_SCALER = False
#     print("⚠️ No scaler found, proceeding without scaling")

# # Routes
# @app.route('/', methods=['GET'])
# def home():
#     return jsonify({
#         'message': 'CardioPredict RandomForest API 🚀',
#         'model': 'RandomForestClassifier',
#         'threshold': THRESHOLD,
#         'status': 'active'
#     })

# @app.route('/predict', methods=['POST'])
# def predict():
#     print("\n" + "="*50)
#     print("📥 Received prediction request")

#     if model is None:
#         return jsonify({'error': 'Model not loaded'}), 500

#     data = request.get_json()
#     print(f"📦 Raw data received: {data}")

#     required_keys = [
#         'age', 'gender', 'height', 'weight',
#         'ap_hi', 'ap_lo', 'smoke', 'alco', 'active',
#         'cholesterol_2', 'cholesterol_3',
#         'gluc_2', 'gluc_3'
#     ]

#     missing = [k for k in required_keys if k not in data]
#     if missing:
#         return jsonify({'error': f'Missing keys: {missing}'}), 400

#     try:
#         X = pd.DataFrame([{
#             'age': float(data['age']),
#             'gender': int(data['gender']),
#             'height': float(data['height']),
#             'weight': float(data['weight']),
#             'ap_hi': float(data['ap_hi']),
#             'ap_lo': float(data['ap_lo']),
#             'smoke': int(data['smoke']),
#             'alco': int(data['alco']),
#             'active': int(data['active']),
#             'cholesterol_2': int(data['cholesterol_2']),
#             'cholesterol_3': int(data['cholesterol_3']),
#             'gluc_2': int(data['gluc_2']),
#             'gluc_3': int(data['gluc_3'])
#         }])

#         print(f"✅ DataFrame created: {X.shape}")

#         # Scaling (only numeric cols)
#         if USE_SCALER:
#             num_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
#             X[num_cols] = scaler.transform(X[num_cols])
#             print("✅ Scaling applied")

#         # Prediction
#         prob = float(model.predict_proba(X)[0][1])
#         prediction = int(prob >= THRESHOLD)

#         # Risk bucket (recommended)
#         if prob < 0.3:
#             risk = "Low"
#         elif prob < 0.6:
#             risk = "Moderate"
#         else:
#             risk = "High"

#         result = {
#             'prediction': prediction,
#             'probability': round(prob, 4),
#             'risk': risk,
#             'threshold': THRESHOLD
#         }

#         print(f"🎯 Result: {result}")
#         print("="*50 + "\n")

#         return jsonify(result)

#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return jsonify({'error': str(e)}), 500

# @app.route('/model-info', methods=['GET'])
# def model_info():
#     return jsonify({
#         'model_type': 'RandomForestClassifier',
#         'threshold': THRESHOLD,
#         'features': [
#             'age', 'gender', 'height', 'weight',
#             'ap_hi', 'ap_lo', 'smoke', 'alco', 'active',
#             'cholesterol_2', 'cholesterol_3', 'gluc_2', 'gluc_3'
#         ],
#         'scaler': 'StandardScaler' if USE_SCALER else 'None'
#     })

# # Run server
# if __name__ == '__main__':
#     print("\n🚀 CardioPredict RandomForest API Starting...")
#     print(f"🌐 http://localhost:5000 | Threshold = {THRESHOLD}")
#     app.run(debug=True, port=5000)


# # Path to frontend
# FRONTEND_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')

# # Serve main page
# @app.route('/')
# def serve_frontend():
#     return send_from_directory(FRONTEND_FOLDER, 'index.html')

# # Serve static files
# @app.route('/<path:path>')
# def serve_static(path):
#     return send_from_directory(FRONTEND_FOLDER, path)

# # API home
# @app.route('/api', methods=['GET'])
# def home_api():
#     return jsonify({
#         'message': 'CardioPredict RandomForest API 🚀',
#         'model': 'RandomForestClassifier',
#         'threshold': THRESHOLD,
#         'status': 'active'
#     })
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

THRESHOLD = 0.4

# Load model
print("📦 Loading RandomForest model...")
try:
    model = joblib.load("rf_model.pkl")
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None

# Load scaler
try:
    scaler = joblib.load("scaler.pkl")
    USE_SCALER = True
    print("✅ Scaler loaded")
except:
    scaler = None
    USE_SCALER = False
    print("⚠️ No scaler found")

# ---------------- API ROUTES ---------------- #

@app.route("/api", methods=["GET"])
def api_home():
    return jsonify({
        "message": "CardioPredict API 🚀",
        "model": "RandomForestClassifier",
        "threshold": THRESHOLD
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()

    required = [
        'age','gender','height','weight',
        'ap_hi','ap_lo','smoke','alco','active',
        'cholesterol_2','cholesterol_3','gluc_2','gluc_3'
    ]

    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields {missing}"}), 400

    X = pd.DataFrame([data])

    if USE_SCALER:
        num_cols = ['age','height','weight','ap_hi','ap_lo']
        X[num_cols] = scaler.transform(X[num_cols])

    prob = float(model.predict_proba(X)[0][1])
    prediction = int(prob >= THRESHOLD)

    risk = "Low" if prob < 0.3 else "Moderate" if prob < 0.6 else "High"

    return jsonify({
        "prediction": prediction,
        "probability": round(prob, 4),
        "risk": risk
    })

# ---------------- FRONTEND ---------------- #

@app.route("/")
def frontend():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("frontend", path)

# ---------------- RUN ---------------- #

# if __name__ == "__main__":
#     app.run(
#         host="0.0.0.0",
#         port=int(os.environ.get("PORT", 10000))
#     )
