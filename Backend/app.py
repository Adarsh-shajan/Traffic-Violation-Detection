from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    Response,
    jsonify
)
import os
import csv
import cv2
import time
from threading import Thread, Lock, Event
from werkzeug.utils import secure_filename
from detection import process_video

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
VIOLATION_FOLDER = "violation_images"
CSV_FILE = "violations.csv"

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables for background thread and live video streaming
detection_thread = None
latest_frame = None
frame_lock = Lock()
stop_event = Event() # The flag to stop the thread gracefully

# Create folders if missing
for folder in [UPLOAD_FOLDER, VIOLATION_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Disable caching
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/')
def index():
    global detection_thread
    
    is_processing = detection_thread is not None and detection_thread.is_alive()
    violations = []
    
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "Track ID", "Plate Number", "Violation", "Image Path"])

    with open(CSV_FILE, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = row.get("Image Path", "")
            if not image_path:
                continue

            image_name = os.path.basename(image_path)
            full_image_path = os.path.join(VIOLATION_FOLDER, image_name)

            if not os.path.exists(full_image_path):
                continue

            violations.append({
                "time": row.get("Time", ""),
                "track_id": row.get("Track ID", ""),
                "plate": row.get("Plate Number", ""),
                "violation": row.get("Violation", ""),
                "image": image_name
            })

    violations.reverse()
    
    return render_template(
        "index.html", 
        violations=violations, 
        processing=is_processing
    )

# -----------------------------------
# VIDEO STREAMING LOGIC
# -----------------------------------
def update_frame(frame):
    global latest_frame
    with frame_lock:
        latest_frame = frame.copy()

def generate_frames():
    global latest_frame
    while detection_thread is not None and detection_thread.is_alive():
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        with frame_lock:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------
# STATUS & QUIT LOGIC
# -----------------------------------
@app.route('/status')
def status():
    global detection_thread
    is_processing = detection_thread is not None and detection_thread.is_alive()
    return jsonify({"processing": is_processing})

@app.route('/quit', methods=['POST'])
def quit_detection():
    """Triggered by the UI to stop the AI."""
    global stop_event
    stop_event.set() # Signal the thread to break its loop
    time.sleep(0.5)  # Give the thread a split second to finish cleaning up
    return jsonify({"status": "stopped"})

# -----------------------------------
# UPLOAD AND PROCESSING
# -----------------------------------
@app.route('/upload', methods=['POST'])
def upload_video():
    global detection_thread, latest_frame, stop_event

    if 'video' not in request.files:
        return redirect(url_for('index'))

    uploaded_file = request.files['video']
    if uploaded_file.filename == '':
        return redirect(url_for('index'))

    for file_name in os.listdir(VIOLATION_FOLDER):
        file_path = os.path.join(VIOLATION_FOLDER, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    if os.path.exists(CSV_FILE):
        os.remove(CSV_FILE)

    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Track ID", "Plate Number", "Violation", "Image Path"])

    latest_frame = None
    stop_event.clear() # Reset the stop flag before starting a new run

    filename = secure_filename(uploaded_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(video_path)

    # Pass the stop_event into the AI processor
    detection_thread = Thread(target=process_video, args=(video_path, update_frame, stop_event))
    detection_thread.start()

    return redirect(url_for('index'))

@app.route('/violation_images/<filename>')
def violation_image(filename):
    return send_from_directory(VIOLATION_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)