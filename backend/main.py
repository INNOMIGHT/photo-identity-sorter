import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from utils.face_cluster import cluster_faces

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

# Static file mounts
app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Enable CORS for frontend (GitHub Pages etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(request: Request, files: List[UploadFile] = File(...)):
    image_paths = []

    # Save uploaded images
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        image_paths.append(file_path)

    # Run face clustering
    clusters = cluster_faces(image_paths)

    # Dynamic base URL (works for Railway / custom domain / local)
    base_url = str(request.base_url).rstrip("/")

    result = {
        person: [
            f"{base_url}/images/{os.path.basename(p)}"
            for p in imgs
        ]
        for person, imgs in clusters.items()
    }

    # Clean up uploaded images after processing
    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return JSONResponse({"clusters": result})