import numpy as np
from ultralytics import YOLO
import cv2

from util import get_car, read_license_plate, write_csv

from sort.sort.sort import *

motion_tracker = Sort()

results = {}

pretrained_model_path = "pretrained_model/yolov8n.pt"
detector_model_path = "runs/last/last.pt"
video_path = "2103099-uhd_3840_2160_30fps.mp4"

vehicles_model = YOLO(pretrained_model_path) # For detecting vehicles
license_model = YOLO(detector_model_path) # For detecting license plate

cap = cv2.VideoCapture(video_path)

vehicles = [2, 3, 5, 7]

frm_no = -1

ret = True
while ret:
    frm_no += 1
    ret, frame = cap.read()
    if ret:

        results[frm_no] = {}

        # Detect Vehicles
        detections = vehicles_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Track Vehicles
        track_ids = motion_tracker.update(np.asarray(detections_))

        # Detect license plates
        license_plates = license_model(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    results[frm_no][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]
                        },
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }

write_csv(results, 'test.csv')






