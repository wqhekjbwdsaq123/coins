# 🪙 AI 글로벌 동전 분류기 (Coin Classifier)

🌐 **웹 서비스(체험하기):** [https://d25tbzepfgpjyjvislpua3.streamlit.app/](https://d25tbzepfgpjyjvislpua3.streamlit.app/)

카메라로 찍은 이미지에서 동전을 인식하여 전세계 211종의 통화 중 어느 나라 동전인지 자동으로 분류하고, 현재 환율(2026년 2월 기준)을 적용하여 원화(KRW) 기준 총액을 계산해 주는 AI 서비스입니다. YOLOv8 객체 인식 모델 기반으로 제작되었습니다.

---

## 🚀 웹 서비스 이용 방법

위의 **[웹 서비스 링크]**에 접속하여 바로 기능을 사용해 볼 수 있습니다!

1. 서비스 접속 후, 동전이 포함된 사진을 업로드합니다.
2. 좌측 사이드바에서 `최소 반지름`, `원형도 임계치`, `신뢰도 임계치` 등의 파라미터를 조절할 수 있습니다 (기본값으로도 잘 작동합니다).
3. **[🔍 분류 시작]** 버튼을 누르면 AI가 인식한 동전들의 이름, 개수, 그리고 **한화 기준 총액**을 계산하여 보여줍니다.

## 🌟 주요 특징
- **강건한 인식**: 정방향 및 역방향 이미지 이진화 처리 기법을 결합하여, 어두운 배경이나 밝은 배경에서도 잘 감지합니다.
- **211종 통화 지원**: 다양한 국가의 동전 데이터를 학습하여 높은 분류 능력을 제공합니다.
- **실시간 금액 환산**: 검출된 각 동전의 액면가와 환율을 적용하여, 총 얼마인지 한 번에 보여줍니다.

---

## 🛠 직접 모델을 학습하고 싶은 경우 (Google Colab)

웹 서비스가 아닌, 이 모델을 직접 학습해보고 싶거나 Colab 코드를 살펴보고 싶다면 아래 가이드를 참고하세요.

<a href="https://colab.research.google.com/github/wqhekjbwdsaq123/ML_pr/tree/main/coins/동전분류기.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

1. 위쪽의 **Open In Colab** 배지를 클릭하여 노트북을 엽니다.
2. Colab 환경에서 `런타임 > 런타임 유형 변경` 메뉴를 클릭하고 **T4 GPU**를 선택합니다.
3. 데이터 및 학습된 모델(가중치)은 사용자의 Google Drive에 저장하여 불러오게 설정되어 있습니다. 
   - `MyDrive/coins/data` 위치에 Kaggle 데이터셋을 압축 해제해 주세요.
     *(Kaggle 데이터셋: [coin-images by wanderdust](https://www.kaggle.com/datasets/wanderdust/coin-images))*
   - 테스트할 이미지는 `MyDrive/coins/test_scene.jpg`에 넣어주세요.
4. 노트북의 모든 셀을 순서대로 실행합니다 (`Ctrl + F9`). 실행 중 Google Drive 연동을 허용해야 합니다.

## 💻 로컬 환경 실행
로컬 환경에서 코드를 실행하려면 파이썬 의존성을 설치해야 합니다.
```bash
pip install -r requirements.txt
python app.py  # (Gradio 기반 UI)
# 또는
streamlit run streamlit_app.py  # (Streamlit 기반 UI)
```
> 로컬 실행 시 상단의 경로 설정(`CONFIG`)에서 모델 및 데이터 경로를 환경에 맞게 수정해 주세요.

## 📚 사용된 오픈소스 및 기술
- Web Framework: **Streamlit**, **Gradio**
- Deep Learning: **Ultralytics (YOLOv8)**
- Computer Vision: **OpenCV**, **NumPy**
