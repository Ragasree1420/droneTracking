import os
import shutil
import subprocess
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Drone Detection & Tracking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Allowed: MP4, AVI, MOV")

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    saved_name = f"video_{ts}{ext}"
    save_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(save_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    return {"filename": saved_name, "path": save_path}


@app.post("/process")
async def process_video(filename: str = Form(...)):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(input_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    out_name = os.path.splitext(filename)[0] + "_processed.mp4"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    script_path = os.path.join(BASE_DIR, 'demo_detect_track.py')
    if not os.path.isfile(script_path):
        raise HTTPException(status_code=500, detail="Processing script not found")

    # Call the script non-interactively, no GUI, with input/output paths
    try:
        python_exec = os.path.join(BASE_DIR, 'drone_env', 'Scripts', 'python.exe') if os.name == 'nt' else 'python'
        result = subprocess.run(
            [
                python_exec,
                script_path,
                '--input', input_path,
                '--output', output_path,
                '--no-gui'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {e}")

    metrics_path = output_path + '.json'
    response = {
        "output": out_name,
        "metrics": None,
        "stdout": result.stdout[-2000:] if result.stdout else "",
        "stderr": result.stderr[-2000:] if result.stderr else ""
    }
    if os.path.isfile(metrics_path):
        # return only the filename to avoid client path parsing issues across OSes
        response["metrics"] = os.path.basename(metrics_path)

    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Processing did not produce output video")

    return JSONResponse(response)


@app.get("/download/video/{name}")
async def download_video(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="video/mp4", filename=name)


@app.get("/download/metrics/{name}")
async def download_metrics(name: str):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="application/json", filename=name)


@app.get("/")
async def root():
    return {"status": "ok"}

