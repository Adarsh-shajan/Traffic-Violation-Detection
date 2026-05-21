from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import re
import time
import csv
import os

def letter_by_letter_correction(text):
    if len(text) < 7 or len(text) > 10:
        return text

    char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'A': '4'}
    num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G', '4': 'A'}
    processed = list(text)

    for i in range(min(2, len(processed))):
        if processed[i] in num_to_char:
            processed[i] = num_to_char[processed[i]]
    for i in range(2, min(4, len(processed))):
        if processed[i] in char_to_num:
            processed[i] = char_to_num[processed[i]]
    for i in range(max(0, len(processed) - 4), len(processed)):
        if processed[i] in char_to_num:
            processed[i] = char_to_num[processed[i]]
    for i in range(4, max(4, len(processed) - 4)):
        if processed[i] in num_to_char:
            processed[i] = num_to_char[processed[i]]
    return "".join(processed)

ocr = PaddleOCR(use_angle_cls=True, lang='en')

helmet_model = YOLO('runs/detect/helmet_project/helmet_training-6/weights/best.pt')
plate_model = YOLO('runs/detect/number_plate/number_plate_training-2/weights/best.pt')

csv_file = "violations.csv"
images_dir = "violation_images"

if not os.path.exists(images_dir):
    os.makedirs(images_dir)

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time", "Track ID", "Plate Number", "Violation", "Image Path"])

# Added stop_event parameter
def process_video(video_path, frame_callback=None, stop_event=None):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video")
        return

    processed_ids = {}
    saved_entries = set()

    while True:
        
        # -----------------------------------
        # CHECK IF "QUIT" BUTTON WAS CLICKED
        # -----------------------------------
        if stop_event is not None and stop_event.is_set():
            print("Processing safely stopped by user.")
            break

        ret, frame = cap.read()

        if not ret:
            print("Video ended")
            break

        results = helmet_model.track(frame, persist=True)
        annotated = results[0].plot()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                track_id = None

                if box.id is not None:
                    track_id = int(box.id[0])

                if track_id is None:
                    continue

                violation_type = ""
                if cls_id == 5:
                    violation_type = "Driver Without Helmet"
                elif cls_id == 6:
                    violation_type = "Passenger Without Helmet"
                else:
                    continue

                current_time = time.time()
                if track_id in processed_ids:
                    if (current_time - processed_ids[track_id]) < 5:
                        continue

                processed_ids[track_id] = current_time

                cv2.putText(
                    annotated,
                    f"{violation_type} ID:{track_id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                plate_results = plate_model(frame, conf=0.4)

                for pr in plate_results:
                    for pbox in pr.boxes:
                        x1, y1, x2, y2 = map(int, pbox.xyxy[0])
                        plate = frame[y1:y2, x1:x2]

                        if plate.size == 0:
                            continue

                        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                        gray = clahe.apply(gray)
                        gray = cv2.bilateralFilter(gray, 11, 17, 17)

                        result = ocr.ocr(gray)
                        detected_fragments = []

                        if result and result[0]:
                            for word in result[0]:
                                box_pts = word[0]
                                text = word[1][0]
                                confidence = word[1][1]

                                if confidence > 0.4:
                                    center_x = sum([pt[0] for pt in box_pts]) / 4
                                    center_y = sum([pt[1] for pt in box_pts]) / 4
                                    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())

                                    if clean_text:
                                        detected_fragments.append({'text': clean_text, 'x': center_x, 'y': center_y})

                        detected_fragments.sort(key=lambda item: (item['y'] // 20, item['x']))

                        raw_plate_number = "".join([frag['text'] for frag in detected_fragments])
                        plate_number = letter_by_letter_correction(raw_plate_number)

                        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$'
                        if not re.match(pattern, plate_number):
                            continue

                        entry_key = (plate_number, violation_type)
                        if entry_key in saved_entries:
                            continue

                        saved_entries.add(entry_key)
                        print(f"Vehicle Number: {plate_number}")

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        safe_time_str = time.strftime("%Y%m%d_%H%M%S")
                        image_filename = f"{plate_number}_ID{track_id}_{safe_time_str}.jpg"
                        image_path = os.path.join(images_dir, image_filename)

                        cv2.imwrite(image_path, annotated)
                        print(f"Saved Image: {image_filename}")

                        current_time_string = time.strftime("%Y-%m-%d %H:%M:%S")

                        with open(csv_file, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([
                                current_time_string,
                                track_id,
                                plate_number,
                                violation_type,
                                image_path
                            ])

        if frame_callback is not None:
            frame_callback(annotated)

        if len(processed_ids) > 1000:
            processed_ids.clear()

    cap.release()
    print("Video processing ended cleanly.")