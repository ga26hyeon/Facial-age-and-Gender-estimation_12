FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl and CA certificates are needed to fetch model files over HTTPS during
# the image build. The OpenCV runtime libraries are also installed here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Keep model weights out of Git while including them in the built image.
RUN mkdir -p /app/models \
    && curl -fL "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt" -o /app/models/face.prototxt \
    && curl -fL "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb" -o /app/models/face.pb \
    && curl -fL "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt" -o /app/models/age.prototxt \
    && curl -fL "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel" -o /app/models/age.caffemodel \
    && curl -fL "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt" -o /app/models/gender.prototxt \
    && curl -fL "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel" -o /app/models/gender.caffemodel \
    && test -s /app/models/face.prototxt \
    && test -s /app/models/face.pb \
    && test -s /app/models/age.prototxt \
    && test -s /app/models/age.caffemodel \
    && test -s /app/models/gender.prototxt \
    && test -s /app/models/gender.caffemodel

COPY app.py .
COPY static/ ./static/

# Exercise the same loader the app uses; fail the build if any model cannot load.
RUN python -c "import app; app.load_models(); assert app.face_net is not None and app.age_net is not None and app.gender_net is not None; print('All face, age, and gender models loaded successfully')"

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
