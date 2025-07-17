from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import joblib
from ultralytics import YOLO
import traceback

app = FastAPI()

# CORS (Optional for local testing, needed if connecting frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Jaundice Prediction API is live"}

# === LOAD MODELS ===
try:
    print("🔄 Loading models...")
    detection_model = YOLO("sclera_detector.pt")
    prediction_model = joblib.load("jaundice_predicter.pkl")
    print("✅ Models loaded successfully!")
except Exception as e:
    print("❌ Error loading models:", traceback.format_exc())
    raise RuntimeError("Model loading failed")

# === PREDICT ===
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed. Invalid image format.")

        results = detection_model(img)
        print("🔍 YOLO detection complete.")

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                eye_img = img[y1:y2, x1:x2]

                if eye_img.size == 0:
                    raise ValueError("Detected region is empty.")

                eye_img_resized = cv2.resize(eye_img, (64, 64))
                blue_channel = eye_img_resized[:, :, 0]
                mean_blue = np.mean(blue_channel)
                feature = np.array([[mean_blue]])

                print("🧠 Feature shape:", feature.shape)
                prediction = prediction_model.predict(feature)

                return {"prediction": str(prediction[0])}

        return {"error": "No eye detected by YOLO."}

    except Exception as e:
        error_details = traceback.format_exc()
        print("🔥 ERROR TRACEBACK:\n", error_details)
        return {
            "error": str(e),
            "debug": error_details
        }
