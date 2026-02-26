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

# 3. 통화별 한화(KRW) 환율 매핑 (2026년 2월 기준 근사치)
EXCHANGE_RATES = {
    "australian dollar":    900,
    "brazilian real":       250,
    "british pound":       1700,
    "canadian dollar":     1000,
    "chilean peso":           1.5,
    "chinese yuan renminbi": 195,
    "czech koruna":          60,
    "danish krone":         190,
    "euro":                1500,
    "hong kong dollar":     175,
    "hungarian forint":       3.7,
    "indian rupee":          16,
    "indonesian rupiah":      0.085,
    "israeli new shekel":   380,
    "japanese yen":           9,
    "korean won":             1,
    "malaysian ringgit":    310,
    "mexican peso":          68,
    "new zealand dollar":   820,
    "norwegian krone":      130,
    "pakistan rupee":         5,
    "philipine peso":        24,
    "polish zloty":         340,
    "russian ruble":         14,
    "singapore dollar":    1020,
    "south african rand":    75,
    "swedish krona":        130,
    "swiss franc":         1550,
    "taiwan dollar":         43,
    "thai baht":             40,
    "turkish lira":          40,
    "us dollar":           1450,
}

def parse_face_value(denomination_str: str) -> float:
    """'1 Cent', '50 Paise', '1 2 Dollar'(=1/2) 같은 형식에서 숫자 액면가를 추출"""
    parts = denomination_str.strip().split()
    if not parts:
        return 0.0
    try:
        # '1 2 Dollar' → 1/2 = 0.5 처리
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]) / int(parts[1])
        return float(parts[0].replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return 0.0

def coin_to_krw(class_str_id: str) -> float:
    """클래스 ID → 한화(KRW) 가치 계산"""
    info = cat_to_name.get(class_str_id, "")
    if not info:
        return 0.0
    parts = [p.strip() for p in info.split(",")]
    if len(parts) < 2:
        return 0.0
    denomination = parts[0]   # e.g. "500 Won"
    currency     = parts[1].lower()  # e.g. "korean won"
    rate = EXCHANGE_RATES.get(currency, 0.0)
    face_value = parse_face_value(denomination)
    return face_value * rate

# 모폴로지 연산 및 전처리를 통해 동전 후보 윤곽을 찾는 함수
def detect_coin_circles(th: np.ndarray, min_radius: int = 35, circularity_thresh: float = 0.7):
    """
    circularity = 4π × area / perimeter²
    완전한 원 = 1.0 / 정사각형 ≈ 0.785 / 불규칙한 노이즈 = 낮은값
    circularity_thresh 이상인 윤곽만 동전으로 인정
    """
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        # 반지름 및 원형도 기준 필터링
        if radius > min_radius and circularity >= circularity_thresh:
            circles.append(((cx, cy), radius))
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
    
    # [1] 전처리 (이진화 및 모폴로지)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, th = cv2.threshold(gray, 0, 255, flag)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # [2] 동전 윤곽 원 탐지 (원형도 필터 포함)
    circles = detect_coin_circles(th, min_radius=35, circularity_thresh=0.7)
    
    if not circles:
        return image, "동전이 감지되지 않았습니다.\n(이미지가 너무 흐리거나 동전이 서로 겹쳐 있으면 감지가 어려울 수 있습니다)"
        
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
    krw_per_class = {}
    result_img = src.copy()
    total_krw = 0.0
    unknown_count = 0
    
    # [5] 예측 결과를 원본 이미지 위에 그리기 및 집계
    CONF_THRESH = 0.4  # 신뢰도 임계치 상향 (0.2 → 0.4)
    for pred, (center, radius) in zip(preds, circles):
        cls_id = int(pred.probs.top1)
        class_str_id = model.names[cls_id]
        readable_name = cat_to_name.get(class_str_id, f"Class {class_str_id}")
        conf = float(pred.probs.top1conf)
        
        if conf < CONF_THRESH:
            readable_name = "Unknown"
            short_name = "알수없음"
            unknown_count += 1
            circle_color = (0, 0, 255)   # 빨간 원 = 신뢰도 낮음
        else:
            short_name = readable_name.split(",")[0] if "," in readable_name else readable_name
            krw_val = coin_to_krw(class_str_id)
            total_krw += krw_val
            krw_per_class[readable_name] = krw_val
            circle_color = (0, 255, 0)   # 초록 원 = 정상 인식
            
        counts[readable_name] += 1
        
        # 이미지에 동전 원과 분류 결과 표기
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(result_img, (cx, cy), radius, circle_color, 3)
        cv2.putText(result_img, short_name, (cx - radius, cy - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, circle_color, 2)
        
    # [6] 텍스트 요약 메시지 생성
    result_text = "🪙 검출된 동전 요약:\n\n"
    for name, count in counts.items():
        if "," in name:
            parts = name.split(",")
            country = parts[2].strip().capitalize() if len(parts) > 2 else ""
            coin_info = f"{parts[0].strip()} [{country}]"
        else:
            coin_info = name
        
        krw_val = krw_per_class.get(name, 0.0)
        krw_each = f"≈ {krw_val:,.0f}원/개" if krw_val > 0 else ""
        result_text += f"  • {coin_info}: {count}개  {krw_each}\n"
    
    result_text += "\n" + "─" * 30 + "\n"
    if unknown_count > 0:
        result_text += f"⚠️  인식 불가 동전: {unknown_count}개\n"
    result_text += "  (※ 환율은 2026년 2월 기준 근사치입니다)"
        
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    # 한화 총액을 별도 Markdown 문자열로 반환
    total_md = f"💰 한화 기준 총액: **{total_krw:,.0f} 원**"
    return result_img_rgb, result_text, total_md

# --- Gradio UI ---
custom_css = """
.gradio-container {
    font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
}
.total-krw {
    font-size: 2.2em;
    font-weight: 800;
    color: #1a6bcc;
    text-align: center;
    padding: 16px;
    background: #eef4ff;
    border-radius: 12px;
    border: 2px solid #b3d0f5;
    margin-top: 8px;
}
"""

with gr.Blocks(title="AI 글로벌 동전 분류기") as demo:
    gr.Markdown("# 🪙 AI 글로벌 동전 분류기 (YOLOv8)")
    gr.Markdown("동전이 포함된 사진을 업로드하면, AI가 각 동전을 전세계 **211종**의 통화 중 하나로 자동 분류하고 **한화 기준 총액**을 계산해 드립니다.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="동전 이미지 업로드")
            submit_btn = gr.Button("🔍 분류 시작", variant="primary", size="lg")
            total_display = gr.Markdown(
                value="💰 한화 기준 총액",
                elem_classes=["total-krw"]
            )
            
        with gr.Column():
            output_image = gr.Image(type="numpy", label="분석 결과 이미지 (라벨 표시)")
            output_text = gr.Textbox(label="분류 결과 상세", lines=12)
            
    submit_btn.click(fn=predict_coins, inputs=input_image, outputs=[output_image, output_text, total_display])
    
    gr.Markdown("### 테스트 예시 이미지")
    gr.Examples(
        examples=[
            "Coin_test/KakaoTalk_20260109_095442227.jpg",
            "Coin_test/KakaoTalk_20260109_095442227_01.jpg",
            "Kor_Foreign_Coin/Coin 1.jpg",
        ],
        inputs=input_image
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft(), css=custom_css)
