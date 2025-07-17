from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
import cv2
import torch
import joblib
from ultralytics import YOLO

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Jaundice Prediction API is live"}


# Load the models
detection_model = YOLO("sclera_detector.pt")
prediction_model = joblib.load("jaundice_predicter.pkl")

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read and decode the uploaded image
    contents = await image.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Step 1: Run detection (YOLO)
    results = detection_model(img)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            eye_img = img[y1:y2, x1:x2]
            eye_img_resized = cv2.resize(eye_img, (64, 64))
            eye_features = eye_img_resized.flatten().reshape(1, -1)

            # Step 2: Run jaundice prediction
            prediction = prediction_model.predict(eye_features)

            return {
                "prediction": str(prediction[0])
            }

    return {"error": "No eye detected"}
