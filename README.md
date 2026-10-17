# Face Estimate Lab

로컬 Windows 컴퓨터에서 이미지 속 얼굴을 검출하고 나이대·성별을 추정하는 FastAPI 웹앱입니다. 일반 노트북 CPU에서 실행하도록 OpenCV DNN과 공개 사전학습 모델을 사용합니다.

> 결과는 참고용 추정치이며 실제 나이·성별을 확정하거나 개인을 식별하지 않습니다. 업로드 이미지는 이 앱 외부로 전송되지 않습니다.

## 요구 사항

- Windows 10/11
- Python 3.10 이상
- 인터넷 연결: Python 패키지와 모델을 처음 준비할 때 필요

## 설치

PowerShell에서 프로젝트 폴더로 이동한 뒤 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

Python이 PATH에 없다면 설치된 Python 실행 파일의 전체 경로로 위 명령을 실행하세요.

## 모델 준비

프로젝트 루트에 `models` 폴더를 만들고 다음 6개 파일을 저장합니다.

- `face.prototxt`
- `face.pb`
- `age.prototxt`
- `age.caffemodel`
- `gender.prototxt`
- `gender.caffemodel`

PowerShell에서 다음 명령으로 공개 저장소에서 준비할 수 있습니다.

```powershell
New-Item -ItemType Directory -Force models
curl.exe -L https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt -o models\face.prototxt
curl.exe -L https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector_uint8.pb -o models\face.pb
curl.exe -L https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt -o models\age.prototxt
curl.exe -L https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel -o models\age.caffemodel
curl.exe -L https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt -o models\gender.prototxt
curl.exe -L https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel -o models\gender.caffemodel
```

모델 가중치는 크기가 크므로 GitHub 저장소에 커밋하지 않는 것을 권장합니다.

## 실행

```powershell
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000/`을 엽니다. 이미지를 선택하거나 드래그한 뒤 `분석 시작`을 누르면 됩니다.

## API 확인

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

`models_ready`가 `True`인지 확인하세요.
