# Face Estimate Lab

FastAPI 웹앱이 얼굴을 검출하고 연령대·성별을 추정합니다. 로컬 CPU에서 실행하도록 OpenCV DNN 얼굴 검출기와 ONNX 사전학습 모델을 사용합니다.

> 결과는 참고용 추정치이며 실제 나이·성별을 확정하거나 개인을 식별하지 않습니다. 업로드 이미지는 앱 외부로 전송하지 않습니다.

## 실행 환경

- Python 3.11
- 고정된 패키지 버전: `requirements.txt`
- 모델 준비 시 인터넷 연결 필요

## 설치 및 모델 준비

PowerShell에서 저장소 폴더로 이동한 뒤 실행합니다.

```powershell
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force models
$revision = 'd8e2c4cafb5e14a258202d8f5176973d25a244df'
curl.exe -fL https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt -o models\face.prototxt
curl.exe -fL https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb -o models\face.pb
curl.exe -fL "https://huggingface.co/snakajima/face-lens/resolve/$revision/student.onnx?download=true" -o models\student.onnx
curl.exe -fL "https://huggingface.co/snakajima/face-lens/resolve/$revision/student.json?download=true" -o models\student.json
curl.exe -fL "https://huggingface.co/snakajima/face-lens/resolve/$revision/ATTRIBUTION.csv?download=true" -o models\ATTRIBUTION.csv
```

`models/` is excluded from Git. Docker builds download these assets and verify the ONNX, metadata, and attribution file SHA-256 hashes. `ATTRIBUTION.csv` is included in the image as required by the model publisher's attribution guidance.

The age/gender model is [`snakajima/face-lens`](https://huggingface.co/snakajima/face-lens), revision `d8e2c4cafb5e14a258202d8f5176973d25a244df`, licensed CC BY 4.0. Its age and gender heads were trained from FairFace human labels. The implementation keeps the existing OpenCV detector and uses its face box to create the model's expanded, vertically shifted crop, followed by the model metadata's RGB mean/std normalization.

## 실행

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. Upload a photo or drag it onto the page, then choose **분석 시작**.

## API 확인

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

`status` should be `ok` and `models_ready` should be `True`. In Docker deployments, the response also includes `app_revision` (the CI commit SHA), `model_revision`, and `face_crop_scale`, so an update can be identified at the running endpoint. The upload endpoint remains `POST /api/analyze` and returns face boxes, gender/age labels, and confidence values.

## 모델 평가

`evaluation/` contains the 100 FairFace validation images, a fixed split manifest, and before/after results. The 30-row tuning set is separate from the 70-row held-out final set. Details, data source/license, and limitations are in [evaluation/RESULTS.md](evaluation/RESULTS.md).

## CI/CD

Pushes build and check the Docker image, including model download, SHA verification, model loading, and the HTTP health endpoint. The `main` CI workflow uploads the verified image to Docker Hub; after a successful upload, the Windows self-hosted deployment workflow starts a candidate behind a temporary Caddy proxy, checks the exact app and model revision, and switches the stable local endpoint `http://127.0.0.1:8081`. The existing service at port 8000 is left untouched. Future updates use Caddy's graceful reload, record timestamped health/revision observations during the switch, allow a drain period, and only then stop the prior backend. If candidate validation fails, the active route and prior backend are retained. Set the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` repository secrets and keep a Docker Engine available to the Windows runner.
