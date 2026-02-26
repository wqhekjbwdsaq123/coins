# 자동 동전 분류기 (Coin Classifier)

🌐 **서비스 링크:** [https://d25tbzepfgpjyjvislpua3.streamlit.app/](https://d25tbzepfgpjyjvislpua3.streamlit.app/)

<a href="https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPOSITORY/blob/main/동전분류기.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

카메라로 찍은 이미지에서 동전을 인식하여 어떤 나라의 동전인지 분류하고, 원화 기준 총액을 계산하는 AI 프로젝트입니다. YOLOv8 객체 인식 모델을 기반으로 학습되었습니다.

## 🌟 주요 기능
- OpenCV를 활용한 이미지 전처리 및 동전 영역 검출
- YOLOv8 신경망 모델을 이용한 211종의 국가별 동전 분류
- 인식된 동전의 가치를 원화 환산하여 총액 계산

## 📁 데이터셋
동전 이미지 데이터셋은 Kaggle에서 제공하는 방대한 글로벌 동전 데이터를 기반으로 합니다.
- **Kaggle Link:** [coin-images by wanderdust](https://www.kaggle.com/datasets/wanderdust/coin-images)

## 🚀 실행 안내 (Google Colab 기준)

이 프로젝트는 **Google Colab** 환경에서 GPU를 사용하여 원활하게 실행되도록 구성되어 있습니다.

1. 위쪽의 **Open In Colab** 배지를 클릭하여 노트북을 엽니다.
2. Colab 환경에서 `런타임 > 런타임 유형 변경` 메뉴를 클릭하고 **T4 GPU**를 선택합니다.
3. 데이터 및 학습된 모델(가중치)은 사용자의 Google Drive에 저장하여 불러오게 설정되어 있습니다. 
   - `MyDrive/coins/data` 위치에 Kaggle 데이터셋을 압축 해제해 주세요.
   - `MyDrive/yolov8_project/coins_cls/weights/best.pt` 경로에 학습이 완료된 가중치 파일을 배치해 주세요.
   - 테스트할 이미지는 `MyDrive/coins/test_scene.jpg`에 넣어주세요.
4. 노트북의 모든 셀을 순서대로 실행합니다 (`Ctrl + F9`). 실행 중 Google Drive 연동을 허용해야 합니다.

## 🛠 사용된 기술
- Python 3
- OpenCV
- Ultralytics (YOLOv8)
- Google Colab / Google Drive 연동

---

> **참고:** 로컬 환경에서 실행하려면 상단의 경로 설정(`CONFIG`)에서 `BASE_DIR`과 모델 및 데이터 경로를 로컬 환경에 맞게 수정하시고, `pip install -r requirements.txt` 명령어로 필요 패키지를 설치해 주세요.
