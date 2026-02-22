import os
import shutil
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

face_model = None

def get_face_model():
    global face_model
    if face_model is None:
        face_model = FaceAnalysis(providers=["CPUExecutionProvider"])
        face_model.prepare(ctx_id=0, det_size=(640, 640))
    return face_model

def reset_output():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_blurry(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("Blur score:", variance)
    return variance < threshold


def save_results(groups):
    for person, images in groups.items():
        person_dir = os.path.join(OUTPUT_DIR, person)
        os.makedirs(person_dir, exist_ok=True)

        for img_path in images:
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(person_dir, os.path.basename(img_path)))


def cluster_faces(image_paths):

    model = get_face_model()
    faces = model.get(img)
    reset_output()   # ✅ AUTO CLEAN

    face_data = []

    for path in image_paths:

        img = cv2.imread(path)
        if img is None:
            continue

        faces = face_model.get(img)

        if len(faces) == 0:
            print("No faces:", path)
            continue

        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        for face in faces:

            if face.det_score < 0.55:     # slightly relaxed
                print("Rejected (low confidence):", face.det_score)
                continue


            x1, y1, x2, y2 = face.bbox.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            w, h = x2 - x1, y2 - y1
            area = w * h

            if area < 0.005 * img_area:   # ✅ RELAXED (very important)
                print("Rejected (tiny face):", area)
                continue

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            if is_blurry(crop):
                print("Rejected (blurry)")
                continue

            face_data.append({
                "image": path,
                "embedding": face.embedding
            })

    if len(face_data) == 0:
        return {}

    embeddings = np.array([f["embedding"] for f in face_data])

    print("Total embeddings:", len(embeddings))

    labels = DBSCAN(
        eps=0.55,
        min_samples=1,
        metric="cosine"
    ).fit_predict(embeddings)

    groups = {}

    for label, record in zip(labels, face_data):
        groups.setdefault(f"Person_{label}", set()).add(record["image"])

    groups = {k: list(v) for k, v in groups.items()}

    save_results(groups)

    return groups