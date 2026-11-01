import json
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
STATIC = ROOT / "static"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

app = FastAPI(title="Face Estimate Lab", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

FACE_LENS_REVISION = "d8e2c4cafb5e14a258202d8f5176973d25a244df"
MODEL_URLS = {
    "face.prototxt": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "face.pb": "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb",
    "student.onnx": f"https://huggingface.co/snakajima/face-lens/resolve/{FACE_LENS_REVISION}/student.onnx",
    "student.json": f"https://huggingface.co/snakajima/face-lens/resolve/{FACE_LENS_REVISION}/student.json",
}

face_net = attribute_session = attribute_metadata = None
_model_lock = Lock()
_inference_lock = Lock()


def ensure_models() -> None:
    missing = [name for name in MODEL_URLS if not (MODELS / name).is_file()]
    if missing:
        raise RuntimeError("필요한 모델 파일이 없습니다: " + ", ".join(missing))


def load_models() -> None:
    global face_net, attribute_session, attribute_metadata
    if face_net is not None and attribute_session is not None and attribute_metadata is not None:
        return
    with _model_lock:
        if face_net is not None and attribute_session is not None and attribute_metadata is not None:
            return
        ensure_models()
        # np.fromfile keeps the Windows app usable under non-ASCII workspace paths.
        read_bytes = lambda name: np.fromfile(MODELS / name, dtype=np.uint8)
        loaded_face_net = cv2.dnn.readNetFromTensorflow(
            read_bytes("face.pb"), read_bytes("face.prototxt")
        )
        with (MODELS / "student.json").open(encoding="utf-8") as metadata_file:
            loaded_metadata = json.load(metadata_file)
        loaded_session = ort.InferenceSession(
            str(MODELS / "student.onnx"), providers=["CPUExecutionProvider"]
        )
        heads = loaded_metadata.get("heads", {})
        if loaded_session.get_inputs()[0].name != "input":
            raise RuntimeError("ONNX 모델 입력 이름이 예상한 input과 다릅니다.")
        if not heads.get("gender") or not heads.get("age"):
            raise RuntimeError("ONNX 모델의 성별/연령대 라벨이 없습니다.")
        output_names = {output.name for output in loaded_session.get_outputs()}
        if not {"gender", "age"}.issubset(output_names):
            raise RuntimeError("ONNX 모델에 gender/age 출력이 없습니다.")
        face_net, attribute_session, attribute_metadata = (
            loaded_face_net,
            loaded_session,
            loaded_metadata,
        )


def _softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    values = np.exp(values - np.max(values))
    return values / values.sum()


def _attribute_input(image: np.ndarray, box: tuple[float, int, int, int, int]) -> np.ndarray:
    _, x1, y1, x2, y2 = box
    face_width, face_height = x2 - x1, y2 - y1
    crop_settings = attribute_metadata["crop"]
    size = max(face_width, face_height) * float(crop_settings["scale"])
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2 + face_height * float(crop_settings["shiftY"])
    left, top = center_x - size / 2, center_y - size / 2
    input_size = int(attribute_metadata["inputSize"])
    scale = input_size / size
    transform = np.array(
        [[scale, 0, -left * scale], [0, scale, -top * scale]], dtype=np.float32
    )
    crop = cv2.warpAffine(
        image,
        transform,
        (input_size, input_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(attribute_metadata["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(attribute_metadata["std"], dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))[None, ...].astype(np.float32)


def analyze(image: np.ndarray) -> list[dict[str, Any]]:
    load_models()
    # OpenCV DNN's setInput/forward pair is stateful; serialize it across workers.
    with _inference_lock:
        height, width = image.shape[:2]
        detector_input = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300), (104, 117, 123), False, False
        )
        face_net.setInput(detector_input)
        detections = face_net.forward()
        candidates: list[tuple[float, int, int, int, int]] = []
        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < 0.55:
                continue
            box = detections[0, 0, index, 3:7] * np.array(
                [width, height, width, height]
            )
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 > x1 and y2 > y1:
                candidates.append((confidence, x1, y1, x2, y2))

        results = []
        for box in candidates:
            confidence, x1, y1, x2, y2 = box
            model_input = _attribute_input(image, box)
            gender_scores, age_scores = attribute_session.run(
                ["gender", "age"], {"input": model_input}
            )
            gender_probabilities = _softmax(gender_scores)
            age_probabilities = _softmax(age_scores)
            gender_index = int(gender_probabilities.argmax())
            age_index = int(age_probabilities.argmax())
            gender_head = attribute_metadata["heads"]["gender"][gender_index].lower()
            gender_label = {"female": "여성", "male": "남성"}.get(gender_head)
            if gender_label is None:
                raise RuntimeError(f"지원하지 않는 성별 클래스입니다: {gender_head}")
            age_range = attribute_metadata["heads"]["age"][age_index]
            results.append(
                {
                    "box": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                    },
                    "confidence": round(confidence * 100, 1),
                    "gender": gender_label,
                    "gender_confidence": round(float(gender_probabilities[gender_index]) * 100, 1),
                    "age_range": age_range,
                    "age_confidence": round(float(age_probabilities[age_index]) * 100, 1),
                }
            )
        return results


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    try:
        load_models()
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "models_ready": False, "error": str(exc)},
            status_code=503,
        )
    return {
        "status": "ok",
        "models_ready": True,
        "attribute_model": "snakajima/face-lens",
        "attribute_model_revision": FACE_LENS_REVISION,
    }


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "이미지 파일만 업로드할 수 있습니다."}, status_code=400)
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        return JSONResponse({"error": "파일 크기는 10MB 이하여야 합니다."}, status_code=413)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "이미지를 읽을 수 없습니다."}, status_code=400)
    try:
        from fastapi.concurrency import run_in_threadpool

        faces = await run_in_threadpool(analyze, image)
    except Exception as exc:
        return JSONResponse(
            {"error": f"모델 준비 또는 분석 중 오류가 발생했습니다: {exc}"}, status_code=503
        )
    return {
        "filename": file.filename,
        "faces": faces,
        "notice": "결과는 사전학습 모델의 추정치이며 실제 나이·성별을 확정하지 않습니다.",
    }
