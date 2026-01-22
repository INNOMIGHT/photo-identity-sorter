from backend.utils.face_preprocess import extract_good_faces
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_distances
import numpy as np


def merge_close_clusters(embeddings, labels, merge_threshold=0.35):
    unique_labels = [l for l in set(labels) if l != -1]

    centroids = {}
    for label in unique_labels:
        centroids[label] = np.mean(embeddings[labels == label], axis=0)

    merged = {}
    used = set()

    for i in unique_labels:
        if i in used:
            continue
        merged[i] = [i]
        used.add(i)

        for j in unique_labels:
            if j in used:
                continue
            dist = cosine_distances([centroids[i]], [centroids[j]])[0][0]
            if dist < merge_threshold:
                merged[i].append(j)
                used.add(j)

    return merged


def cluster_faces(image_paths):
    face_data = []

    for path in image_paths:
        good_faces = extract_good_faces(path)

        for face_img in good_faces:
            rep = DeepFace.represent(
                img_path = face_img,
                model_name="ArcFace",
                detector_backend="skip",   
                enforce_detection=False,
                normalization="ArcFace"
            )[0]

            face_data.append({
                "image": path,
                "embedding": rep["embedding"]
            })

    embeddings = np.array([f["embedding"] for f in face_data])


    print("Cosine distance matrix:")
    print(cosine_distances(embeddings))

    clustering = DBSCAN(eps=0.30, min_samples=1, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    print("Cosine distance matrix:")
    print(cosine_distances(embeddings))

    print_cluster_stats(embeddings, labels)

    # Step 1: Merge close identity clusters (handle same-person split)
    merged_map = merge_close_clusters(embeddings, labels)

    # Step 2: Build identity -> image set mapping (multi-label)
    final_groups = {}

    for new_id, old_ids in merged_map.items():
        person_key = f"Person_{new_id}"
        final_groups[person_key] = set()

        for oid in old_ids:
            for record, lab in zip(face_data, labels):
                if lab == oid:
                    final_groups[person_key].add(record["image"])

    # Convert sets to lists for JSON serialization
    final_groups = {k: list(v) for k, v in final_groups.items()}

    return final_groups


def print_cluster_stats(embeddings, labels):
    unique = set(labels)
    for i in unique:
        for j in unique:
            if i >= j or i == -1 or j == -1:
                continue
            ci = np.mean(embeddings[labels == i], axis=0)
            cj = np.mean(embeddings[labels == j], axis=0)
            dist = cosine_distances([ci], [cj])[0][0]
            print(f"Centroid distance Person_{i} vs Person_{j}: {dist:.3f}")
