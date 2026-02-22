import cv2
import numpy as np

def is_blurry(image, threshold=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("Blur score:", variance)
    return variance < threshold


def extract_good_faces(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return []

    # No preprocessing needed anymore
    return [img]