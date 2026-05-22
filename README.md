# Helmet Violation Detection System

## Overview

An intelligent traffic monitoring system built using Python, Flask, YOLO, and PaddleOCR to detect motorcycle helmet violations and extract vehicle number plates in real time.

The system processes uploaded traffic videos, detects riders without helmets, identifies number plates using OCR, stores violation records, and displays the results through a live web dashboard.

---

# Features

* Real-time video processing
* Helmet violation detection
* Rider tracking using YOLO tracking
* Number plate detection
* OCR-based plate recognition
* Automatic image evidence saving
* CSV-based violation logging
* Duplicate entry prevention
* Live video streaming in Flask UI
* Web dashboard for viewing violations

---

# Technologies Used

## Backend

* Python
* Flask

## AI / Computer Vision

* YOLO (Ultralytics)
* PaddleOCR
* OpenCV

## Data Handling

* CSV

---

# Project Structure

```text
project/
│
├── app.py
├── detection.py
├── requirements.txt
├── violations.csv
│
├── uploads/
├── violation_images/
│
├── templates/
│   └── index.html
│
└── static/
     └── style.css
    
```

---

# How It Works

1. User uploads a traffic video through the Flask web interface.
2. YOLO detects motorcycle riders and helmet violations.
3. The system tracks violating riders.
4. Number plates are detected near the violating rider.
5. PaddleOCR extracts plate text.
6. OCR text is cleaned and validated.
7. Violation evidence images are saved.
8. Violation details are stored in a CSV file.
9. Results are displayed in the dashboard.

---

# Installation

## Clone Repository

```bash
git clone --depth 1 https://github.com/Adarsh-shajan/Traffic-Violation-Detection.git
cd .\Traffic-Violation-Detection\    
```


## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```


## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Model Files

Place trained YOLO model weights inside:

```text
runs/detect/
```

Example:

```text
runs/detect/helmet_project/helmet_training-6/weights/best.pt
runs/detect/number_plate/number_plate_training-2/weights/best.pt
```

---

# Run the Project

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

# Violation Information Stored

The system stores:

* Timestamp
* Track ID
* Vehicle Number Plate
* Violation Type
* Saved Evidence Image

---

# Current Supported Violations

* Driver without helmet
* Passenger without helmet

---

# Future Improvements

* Triple riding detection
* Overspeeding detection
* Traffic signal violation detection
* SQLite/MySQL database integration
* Dashboard analytics
* Email/SMS alerts
* Webcam live monitoring
* Dark mode UI
* Multi-camera support

---

# Screenshots
Dashboard
<img width="1919" height="948" alt="Screenshot 2026-05-22 100902" src="https://github.com/user-attachments/assets/edc89e33-1574-4d68-91f9-be4a52bdc3be" />
Video procesing
<img width="1918" height="947" alt="Screenshot 2026-05-22 100935" src="https://github.com/user-attachments/assets/a0206dd7-31c5-4ac5-a63c-ee640e13e2f3" />
Results on the dashboard
<img width="1919" height="955" alt="Screenshot 2026-05-22 101001" src="https://github.com/user-attachments/assets/dc94a670-3573-4d10-9335-8481f0d481bb" />
Snap of violation Detection
<img width="1919" height="947" alt="Screenshot 2026-05-22 101012" src="https://github.com/user-attachments/assets/0961c3fb-a6ac-45c3-964c-9cb515bb2b2d" />



---

# Performance Notes

* OCR accuracy depends on plate visibility and video quality.
* Better lighting improves recognition accuracy.
* GPU acceleration is recommended for smoother processing.

---

# Author

Adarsh Shajan

---

# License

This project is for educational and research purposes.

