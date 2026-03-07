import os
import shutil
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load model once globally
face_model = FaceAnalysis(providers=["CPUExecutionProvider"])
face_model.prepare(ctx_id=0, det_size=(320, 320))   # 640 → 320 (much faster)



# Utilities
def reset_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_blurry(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def save_results(groups):
    for person, images in groups.items():
        person_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(person_dir, exist_ok=True)

        for img_path in images:
            if os.path.exists(img_path):
                shutil.copy(
                    img_path,
                    os.path.join(person_dir, os.path.basename(img_path))
                )



# Main clustering pipeline
def cluster_faces(image_paths):

    reset_output()

    face_data = []

    for path in image_paths:

        img = cv2.imread(path)
        if img is None:
            continue

        faces = face_model.get(img)

        if not faces:
            continue

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        for face in faces:

            # Reject low confidence detections
            if face.det_score < 0.55:
                continue

            x1, y1, x2, y2 = face.bbox.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Reject tiny faces (background people)
            if area < 0.005 * img_area:
                continue

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Reject blurry faces
            if is_blurry(crop):
                continue

            face_data.append((path, face.embedding))

    if not face_data:
        return {}

    # --------------------------------------------------
    # Build embedding matrix
    # --------------------------------------------------

    embeddings = np.array([e for _, e in face_data])

    # --------------------------------------------------
    # Cluster faces
    # --------------------------------------------------

    labels = DBSCAN(
        eps=0.55,
        min_samples=1,
        metric="cosine"
    ).fit_predict(embeddings)

    groups = {}

    for label, (img_path, _) in zip(labels, face_data):
        key = f"Person_{label}"

        if key not in groups:
            groups[key] = []

        if img_path not in groups[key]:
            groups[key].append(img_path)

    save_results(groups)

    return groups