from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
from ultralytics import YOLO
import traceback
from utils import prediction_pipeline  # ← Here bro
    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
   allow_origins=["https://lumicare-ai.netlify.app"],  # React frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Jaundice Prediction API is live"}

# === Load Models ===
try:
    print("🔄 Loading models...")
    detection_model = YOLO("sclera_detector.pt")
    prediction_model = joblib.load("jaundice_predicter.pkl")
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Model Load Error:\n", traceback.format_exc())
    raise RuntimeError("Model loading failed.")

# === Predict Endpoint ===
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed.")

        original, sclera_img, yellow_mask, JI, label = prediction_pipeline(img, detection_model, prediction_model)

        if label in ["No sclera detected", "Sclera region empty"]:
            return {"error": label}

        return {
            "prediction": label,
            "jaundice_index": JI
        }

    except Exception as e:
        print("🔥 Exception Traceback:\n", traceback.format_exc())
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }
