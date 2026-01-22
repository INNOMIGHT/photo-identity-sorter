from typing import List
from fastapi import FastAPI, UploadFile, File
import os
from backend.utils.face_cluster import cluster_faces
from backend.utils.file_utils import save_grouped_photos, clear_folder
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os

UPLOAD_DIR = "backend/uploads"
OUTPUT_DIR = "backend/output"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    image_paths = []

    for file in files:
        file_path = os.path.join("backend/uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        image_paths.append(file_path)

    clusters = cluster_faces(image_paths)
    app.mount("/images", StaticFiles(directory="backend/uploads"), name="images")


    # Convert file paths to URLs
    result = {}
    for person, imgs in clusters.items():
        result[person] = [f"http://127.0.0.1:8000/images/{os.path.basename(p)}" for p in imgs]

    return JSONResponse({"clusters": result})