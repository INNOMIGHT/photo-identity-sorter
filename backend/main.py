import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Request
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://innomight.github.io",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...), request: Request = None):
    image_paths = []

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        image_paths.append(file_path)

    clusters = cluster_faces(image_paths)

    base_url = str(request.base_url).rstrip("/") if request else ""

    result = {
        person: [
            f"{base_url}/images/{os.path.basename(p)}"
            for p in imgs
        ]
        for person, imgs in clusters.items()
    }

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return JSONResponse({"clusters": result})