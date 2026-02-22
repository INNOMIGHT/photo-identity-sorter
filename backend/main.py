import os
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.utils.face_cluster import cluster_faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

# ✅ Mount ONCE (correct)
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

request_base = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    image_paths = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        image_paths.append(file_path)

    clusters = cluster_faces(image_paths)

    # ✅ Correct filename usage
    result = {
        person: [
            f"{request_base}/images/{os.path.basename(p)}"
            for p in imgs
        ]
        for person, imgs in clusters.items()
    }

    return JSONResponse({"clusters": result})