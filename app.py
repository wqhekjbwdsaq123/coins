import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import json
import os

# 1. 모델 라벨 정보 로드 (211개 국가 동전 정보)
try:
    with open("cat_to_name.json", "r", encoding="utf-8") as f:
        cat_to_name = json.load(f)
except Exception as e:
    cat_to_name = {}

# 2. 방금 학습한 YOLOv8 최적 모델(best.pt) 로드
MODEL_PATH = "best.pt"
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = None

# 모폴로지 연산 및 전처리를 통해 동전 후보 윤곽을 찾는 함수
def detect_coin_circles(th: np.ndarray, min_radius: int = 25):
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = [cv2.minEnclosingCircle(c) for c in contours]
    circles = [(center, int(radius)) for center, radius in circles if radius > min_radius]
    circles.sort(key=lambda x: (x[0][0], x[0][1]))
    return circles

# 메인 추론 함수
def predict_coins(image):
    if model is None:
        return image, "❗ 오류: 'best.pt' 모델 파일을 찾을 수 없습니다."
        
    if image is None:
        return None, "이미지를 업로드해주세요."

    # Gradio는 이미지를 RGB 형태로 넘겨줍니다. OpenCV용으로 BGR 변환
    src = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # [1] 전처리 (이진화 빛 모폴로지)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, th = cv2.threshold(gray, 0, 255, flag)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # [2] 동전 윤곽 원 탐지
    circles = detect_coin_circles(th, min_radius=25)
    
    if not circles:
        return image, "동전이 감지되지 않았습니다."
        
    # [3] 각각의 동전을 이미지에서 잘라내기
    coin_imgs = []
    crop_scale = 2.5
    for center, radius in circles:
        r = max(2, int(radius * crop_scale))
        cen = (r // 2, r // 2)
        mask = np.zeros((r, r, 3), np.uint8)
        cv2.circle(mask, cen, radius, (255, 255, 255), cv2.FILLED)
        coin = cv2.getRectSubPix(src, (r, r), center)
        coin = cv2.bitwise_and(coin, mask)
        coin_imgs.append(coin)
    
    # [4] 모델 예측 수행
    preds = model.predict(coin_imgs, imgsz=224, verbose=False)
    
    counts = Counter()
    result_img = src.copy()
    
    # [5] 예측 결과를 원본 이미지 위에 그리기 및 텍스트 집계
    for i, (pred, (center, radius)) in enumerate(zip(preds, circles)):
        cls_id = int(pred.probs.top1)
        # model.names는 '1', '2' 형식의 문자열 클래스명 
        class_str_id = model.names[cls_id] 
        # cat_to_name.json에서 상세 정보 매핑
        readable_name = cat_to_name.get(class_str_id, f"Class {class_str_id}")
        
        conf = float(pred.probs.top1conf)
        
        # 신뢰도 임계치
        if conf < 0.2:
            readable_name = "Unknown"
            short_name = "알수없음"
        else:
            short_name = readable_name.split(",")[0] if "," in readable_name else readable_name
            
        counts[readable_name] += 1
        
        # 이미지에 동전 원과 분류 결과 표기 (초록색 원, 빨간색 글자)
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(result_img, (cx, cy), radius, (0, 255, 0), 3)
        cv2.putText(result_img, short_name, (cx - radius, cy - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
    # [6] 텍스트 요약 메시지 생성
    result_text = "🪙 찰칵! 검출된 동전 요약 정보:\n\n"
    for name, count in counts.items():
        if "," in name:
            parts = name.split(",")
            # 형식 조립 (1 Cent, 호주달러 등)
            country = parts[2].capitalize() if len(parts) > 2 else ""
            coin_info = f"{parts[0]} [{country}]"
        else:
            coin_info = name
            
        result_text += f"- {coin_info}: {count}개 감지됨\n"
        
    # Gradio 출력을 위해 다시 RGB 변환
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    return result_img_rgb, result_text

# --- Gradio UI 스크립트 공간 ---
custom_css = """
.gradio-container {
    font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
}
"""

with gr.Blocks(title="AI 글로벌 동전 분류기", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🪙 AI 글로벌 동전 분류기 (YOLOv8)")
    gr.Markdown("동전이 포함된 사진들을 업로드하면, AI가 각 동전을 전세계 **211종**의 통화 중 하나로 자동 분류하고 테두리를 그려줍니다.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="동전 이미지 업로드")
            submit_btn = gr.Button("🔍 분류 시작", variant="primary", size="lg")
            
        with gr.Column():
            output_image = gr.Image(type="numpy", label="분석 결과 이미지 (라벨 표시)")
            output_text = gr.Textbox(label="분류 결과 통계", lines=10)
            
    # 클릭 시 실행되는 동작 연결
    submit_btn.click(fn=predict_coins, inputs=input_image, outputs=[output_image, output_text])
    
    gr.Markdown("### 테스트 예시 이미지")
    gr.Examples(
        examples=[
            "test_scene.jpg",
            "Coin_test/KakaoTalk_20260109_095442227.jpg"
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
