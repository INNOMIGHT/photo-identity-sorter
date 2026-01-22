import cv2
import numpy as np
from retinaface import RetinaFace

def is_blurry(image, threshold=12):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("Blur score:", variance)
    return variance < threshold

def extract_good_faces(image_path, min_size=80, size_ratio=0.4):
    img = cv2.imread(image_path)
    detections = RetinaFace.detect_faces(img)

    if not isinstance(detections, dict):
        return []

    faces = []
    areas = []

    # First pass: collect areas
    for face in detections.values():
        x1, y1, x2, y2 = face["facial_area"]
        w, h = x2 - x1, y2 - y1
        area = w * h
        areas.append(area)
        faces.append((face, area))

    max_area = max(areas)

    good_faces = []
    for face, area in faces:
        x1, y1, x2, y2 = face["facial_area"]
        w, h = x2 - x1, y2 - y1

        # 1. Relative size filter
        if area < size_ratio * max_area:
            print("Rejected (background face):", area)
            continue

        crop = img[y1:y2, x1:x2]

        # 2. Blur filter
        if is_blurry(crop):
            print("Rejected (blurry)")
            continue

        good_faces.append(crop)

    return good_faces
