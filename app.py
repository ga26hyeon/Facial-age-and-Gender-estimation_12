from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

app = FastAPI(title="Face Estimate Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

MODEL_URLS = {
    "face.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "face.pb": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    "age.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    "gender.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender.caffemodel": "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
}

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
GENDER_BUCKETS = ["여성", "남성"]
MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

face_net = gender_net = age_net = None


def ensure_models() -> None:
    missing = [name for name in MODEL_URLS if not (MODELS / name).exists()]
    if missing:
        raise RuntimeError("모델 파일이 없습니다. models 폴더를 채운 뒤 다시 시도하세요: " + ", ".join(missing))


def load_models() -> None:
    global face_net, gender_net, age_net
    if face_net is not None:
        return
    ensure_models()
    # OpenCV's Windows file loader can reject non-ASCII paths. Read bytes first
    # so the app also works from folders whose names contain Korean characters.
    read_bytes = lambda name: np.fromfile(MODELS / name, dtype=np.uint8)
    face_net = cv2.dnn.readNetFromTensorflow(read_bytes("face.pb"), read_bytes("face.prototxt"))
    gender_net = cv2.dnn.readNetFromCaffe(read_bytes("gender.prototxt"), read_bytes("gender.caffemodel"))
    age_net = cv2.dnn.readNetFromCaffe(read_bytes("age.prototxt"), read_bytes("age.caffemodel"))


def analyze(image: np.ndarray) -> list[dict[str, Any]]:
    load_models()
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 117, 123), False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < 0.55:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        face = image[y1:y2, x1:x2]
        crop = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
        gender_net.setInput(crop)
        gender_scores = gender_net.forward()[0]
        age_net.setInput(crop)
        age_scores = age_net.forward()[0]
        gender_idx = int(gender_scores.argmax())
        age_idx = int(age_scores.argmax())
        results.append({
            "box": {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)},
            "confidence": round(confidence * 100, 1),
            "gender": GENDER_BUCKETS[gender_idx],
            "gender_confidence": round(float(gender_scores[gender_idx]) * 100, 1),
            "age_range": AGE_BUCKETS[age_idx],
            "age_confidence": round(float(age_scores[age_idx]) * 100, 1),
        })
    return results


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "models_ready": all((MODELS / name).exists() for name in MODEL_URLS)}


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "이미지 파일만 업로드할 수 있습니다."}, status_code=400)
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        return JSONResponse({"error": "파일 크기는 10MB 이하만 지원합니다."}, status_code=413)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "이미지를 읽을 수 없습니다."}, status_code=400)
    try:
        faces = analyze(image)
    except Exception as exc:
        return JSONResponse({"error": f"모델 준비 또는 분석 중 오류가 발생했습니다: {exc}"}, status_code=503)
    return {"filename": file.filename, "faces": faces, "notice": "결과는 사전학습 모델의 추정치이며 실제 나이·성별을 확정하지 않습니다."}
