import os
from typing import List
from fastapi import FastAPI, UploadFile, File
from backend.utils.face_cluster import cluster_faces
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

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

    app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

    result = {}
    for person, imgs in clusters.items():
        result[person] = [
            f"http://{os.getenv('PUBLIC_IP', '13.239.35.7')}:8000/images/{os.path.basename(p)}"
            for p in imgs
        ]

    return JSONResponse({"clusters": result})
