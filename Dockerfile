FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl and CA certificates are needed to download the pinned model assets over HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG FACE_LENS_REVISION=d8e2c4cafb5e14a258202d8f5176973d25a244df
RUN mkdir -p /app/models \
    && curl -fL --retry 5 --retry-all-errors "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt" -o /app/models/face.prototxt \
    && curl -fL --retry 5 --retry-all-errors "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb" -o /app/models/face.pb \
    && curl -fL --retry 5 --retry-all-errors "https://huggingface.co/snakajima/face-lens/resolve/${FACE_LENS_REVISION}/student.onnx?download=true" -o /app/models/student.onnx \
    && curl -fL --retry 5 --retry-all-errors "https://huggingface.co/snakajima/face-lens/resolve/${FACE_LENS_REVISION}/student.json?download=true" -o /app/models/student.json \
    && curl -fL --retry 5 --retry-all-errors "https://huggingface.co/snakajima/face-lens/resolve/${FACE_LENS_REVISION}/ATTRIBUTION.csv?download=true" -o /app/models/ATTRIBUTION.csv \
    && echo "2c7405e280a9281c1bd0721f929a14b6da47d24b6cb1193aaf2bd4799d2640da  /app/models/student.onnx" | sha256sum -c - \
    && echo "fdf1ae3136aec4e4026d8bb04cad9d971263b6ae3653d2b376df8ee8e05a5bd5  /app/models/student.json" | sha256sum -c - \
    && echo "ac7cdac012364e6182b4ed6bc269ceec3b3c4a86e32ebda99d4d8a85dfeb3f04  /app/models/ATTRIBUTION.csv" | sha256sum -c - \
    && test -s /app/models/face.prototxt \
    && test -s /app/models/face.pb

COPY app.py .
COPY MODEL_NOTICE.md /app/models/NOTICE.md
COPY static/ ./static/

# Exercise the same loader the app uses; fail the build if either network or metadata fails to load.
RUN python -c "import app; app.load_models(); assert app.face_net is not None and app.attribute_session is not None and app.attribute_metadata is not None; print('Face detector and pinned age/gender ONNX model loaded successfully')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "120"]
