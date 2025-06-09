# 🚗 Traffic Vehicle Detection System

An interactive web application built using **Streamlit** that performs **real-time vehicle detection** from images and videos using the **YOLOv8** object detection model.

---

## 📌 Overview

This app enables users to:
- Upload **images or videos** of traffic scenes
- Detect and classify vehicles such as **cars, trucks, buses, motorcycles, and bicycles**
- Visualize detections with annotated media
- Get **vehicle counts and traffic flow charts**
- Export detection data for further analysis

---

## 🧠 Model Used

- **Primary**: `best.pt` (custom YOLOv8 model)
- **Fallback**: `yolov8m.pt` (pretrained medium YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics))
- **Framework**: [Ultralytics YOLO](https://docs.ultralytics.com/)

---
## 📂 Project Structure
```
vehicle-detection-app/
├── vehicle_detection_app.py     # Main Streamlit application
├── best.pt                      # Optional: Custom YOLOv8 model file
├── requirements.txt             # Required packages
└── README.md                    # Project documentation
```

## 📸 Supported Media

- **Image Formats**: `.jpg`, `.jpeg`, `.png`
- **Video Formats**: `.mp4`, `.avi`, `.mov`

---

## ⚙️ Features

- 🚘 Detect multiple vehicle types
- ⚙️ Adjustable confidence threshold
- 📊 Realtime vehicle count statistics
- 📈 Line chart of traffic flow over time (for videos)
- ⬇️ Download CSV of detection results
- 🧼 Automatic fallback to pretrained model if custom model fails to load

---

## 🧪 Setup Instructions

### 🔁 Clone the Repository

```bash
git clone https://github.com/yourusername/vehicle-detection-app.git
cd vehicle-detection-app
```


### 2. Set up your Python environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run vehicle_detection_app.py
```
## 📦 Requirements

See `requirements.txt` for a full list. Key libraries include:

-`streamlit`
-`opencv-python-headless`
-`numpy`
-`pandas`
-`ultralytics`
-`Pillow`

---
## Project :

Developed by: **Sathya Immanuel Raj B**  
Email: sathyaimmanuelraj@gmail.com  
LinkedIn: [Sathya Immanuel Raj B](https://www.linkedin.com/in/sathya-immanuel-raj-b-43530b303)

