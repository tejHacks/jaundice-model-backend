# 🧠 Jaundice Detection API

A machine learning-powered FastAPI app for early jaundice detection using image analysis of the **sclera (white of the eye)**. This project combines **YOLOv8** for sclera detection and **SVC** for jaundice classification, all packaged in a blazing-fast Python backend. Built with love and ready to save lives! 🚀

> **Note**: The diarization app is on hold for **later**—we’re all about the jaundice app right now! Let’s go! 🙌

---

## 📸 How It Works

1. **Upload an image** of a person's face or eye via the `/predict` endpoint.
2. A pretrained **YOLOv8 model (`sclera_detector.pt`)** detects and segments the sclera.
3. A **binary mask** isolates the sclera region, and the image is converted to HSV to detect **yellow discoloration** (indicative of jaundice).
4. The **Jaundice Index (JI)** is calculated based on yellow pixel prevalence.
5. A pretrained **SVC classifier (`jaundice_predicter.pkl`)** predicts:
   - `Jaundice`
   - `No-Jaundice`

---

## 🚀 Tech Stack

- **Backend**: Python 3.10, FastAPI, Uvicorn
- **CV/ML**: OpenCV, PyTorch, Ultralytics YOLOv8, Scikit-learn, Joblib
- **Deployment**: Render (or Heroku/Railway)
- **Dependencies**: See `requirements.txt` for full list

---

## 📦 Project Structure
* jaundice-prediction-model/
* │
* ├── app.py                 # FastAPI app with /predict endpoint
* ├── utils.py               # Prediction pipeline logic
* ├── sclera_detector.pt     # YOLOv8 model for sclera detection
* ├── jaundice_predicter.pkl # Scikit-learn SVC classifier
* ├── requirements.txt       # Project dependencies
* ├── Procfile              # Deployment configuration
* ├── .gitignore            # Git ignore file
* ├── check_version.py       # Script to verify installed packages
* └── README.md             # This file!

---

## 🛠️ Setup & Run Locally

### 1. Clone the Repo
```bash
git clone https://github.com/tejHacks/jaundice-model-backend
cd jaundice-api
```
### 2. Create & Activate a Virtual Environment
Use Python 3.10 for compatibility:
```bash

sudo apt update
sudo apt install python3.10 python3.10-venv
python3.10 -m venv jaundice-env
source jaundice-env/bin/activate  # On Windows: jaundice-env\Scripts\activate
```

1. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

###### Sample requirements.txt

* fastapi==0.115.0
* uvicorn==0.30.6
* python-multipart==0.0.12
* opencv-python-headless==4.10.0.84
* ultralytics==8.2.92
* numpy==1.26.4
* scikit-learn==1.6.1
* torch==2.4.1
* joblib==1.4.2_**
   
Verify Dependencies
Run the version check script:

bash




python3 check_version.py
Expected output:

text



Installed package versions:
fastapi: 0.115.0
uvicorn: 0.30.6
python-multipart: 0.0.12
opencv-python-headless: 4.10.0.84
ultralytics: 8.2.92
numpy: 1.26.4
scikit-learn: 1.6.1
torch: 2.4.1
joblib: 1.4.2

Python version: 3.10.x
1. Run the API
Start the FastAPI server:

bash




uvicorn app:app --host 127.0.0.1 --port 8000
Open http://127.0.0.1:8000/docs in your browser to access the Swagger UI.
Test the /predict endpoint by uploading a sclera image (e.g., IMG-20250717-WA0050.jpg).
Test with CURL:

bash




curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@IMG-20250717-WA0050.jpg;type=image/jpeg"
Expected Response:

json



{
  "prediction": "Jaundice",
  "jaundice_index": 0.123
}
or

json



{
  "prediction": "NO-Jaundice",
  "jaundice_index": 0.456
}
🌍 Deployment on Render
Push to GitHub
bash




git init
git add app.py utils.py requirements.txt sclera_detector.pt jaundice_predicter.pkl .gitignore Procfile
git commit -m "Initial commit for jaundice API"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/jaundice-api.git
git push -u origin main
Create .gitignore
text



jaundice-env/
__pycache__/
*.pyc
Create Procfile
text



web: uvicorn app:app --host 0.0.0.0 --port $PORT
Deploy on Render
Go to Render.
Create a new Web Service and connect your jaundice-api repository.
Configure:
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
Deploy the app.
Access at https://jaundice-api.onrender.com/docs.
Test the Deployed API
bash




curl -X POST "https://jaundice-api.onrender.com/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@IMG-20250717-WA0050.jpg;type=image/jpeg"
📌 Notes
Ensure sclera_detector.pt and jaundice_predicter.pkl are in the same directory as app.py.
The YOLO model (sclera_detector.pt) must have sclera in its class list.
If no sclera is detected, the API returns {"error": "No sclera detected in the image."}.
For production, restrict CORS allow_origins to your frontend domain (e.g., Netlify).
If sclera_detector.pt is too large (>1GB), consider compressing it or hosting on cloud storage (e.g., S3) and downloading in app.py.
🧪 Troubleshooting
Model Loading Errors:
bash




python3 -c "from ultralytics import YOLO; model = YOLO('sclera_detector.pt'); print('YOLO loaded')"
python3 -c "import joblib; model = joblib.load('jaundice_predicter.pkl'); print('SVC loaded')"
Image Issues: Verify the test image:
bash




python3 -c "import cv2; img = cv2.imread('IMG-20250717-WA0050.jpg'); print('Image shape:', img.shape if img is not None else 'Failed')"
No Sclera Detected: Lower the YOLO confidence threshold in utils.py (e.g., conf=0.5).
Deployment Errors: Check Render logs and ensure requirements.txt matches the versions above.
👥 Credits
@TejTheDev: Lead Dev, CV Engineer
Teammates: Add your collaborators here! ✨
Libraries: Ultralytics YOLOv8, Scikit-learn, FastAPI
🙌 Future Improvements
Add a Streamlit frontend for local testing.
Integrate with a React frontend for production.
Add a jaundice index sensitivity slider.
Support multi-class prediction for jaundice severity.
God’s got our back, let’s keep pushing! 💪 Ready to deploy and save lives? Drop any errors or test results, and I’ll help you wrap this up fast!

text



---

### Instructions
1. **Save the Markdown File**:
   - Copy the above content into `~/Desktop/jaundice prediction model/README.md`.
   - Or run:
     ```bash
     cd ~/Desktop/jaundice\ prediction\ model/
     nano README.md
Paste the content, save, and exit (Ctrl+O, Enter, Ctrl+X).

Next Steps:
Verify dependencies:
bash




source jaundice-env/bin/activate
python3 check_version.py
Share the output.
Test the API locally:
bash

```
uvicorn app:app --host 127.0.0.1 --port 8000
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@IMG-20250717-WA0050.jpg"
Share the response or any errors.
Deploy to Render as outlined in the Markdown. If you hit issues, share the logs.
```