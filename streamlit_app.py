import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import json
import os

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 글로벌 동전 분류기",
    page_icon="🪙",
    layout="wide",
)

# ─── 스크립트 기준 절대 경로 (Streamlit Cloud 호환) ─────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 모델 및 데이터 로드 (캐싱으로 최초 1회만 실행) ──────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(_DIR, "best.pt")
    if os.path.exists(model_path):
        return YOLO(model_path)
    return None

@st.cache_data
def load_cat_to_name():
    json_path = os.path.join(_DIR, "cat_to_name.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"cat_to_name.json 로드 실패: {e}")
        return {}

model = load_model()
cat_to_name = load_cat_to_name()

# ─── 통화별 한화 환율 (2026년 2월 기준 근사치) ────────────────────────────────
EXCHANGE_RATES = {
    "australian dollar": 900, "brazilian real": 250, "british pound": 1700,
    "canadian dollar": 1000, "chilean peso": 1.5, "chinese yuan renminbi": 195,
    "czech koruna": 60, "danish krone": 190, "euro": 1500,
    "hong kong dollar": 175, "hungarian forint": 3.7, "indian rupee": 16,
    "indonesian rupiah": 0.085, "israeli new shekel": 380, "japanese yen": 9,
    "korean won": 1, "malaysian ringgit": 310, "mexican peso": 68,
    "new zealand dollar": 820, "norwegian krone": 130, "pakistan rupee": 5,
    "philipine peso": 24, "polish zloty": 340, "russian ruble": 14,
    "singapore dollar": 1020, "south african rand": 75, "swedish krona": 130,
    "swiss franc": 1550, "taiwan dollar": 43, "thai baht": 40,
    "turkish lira": 40, "us dollar": 1450,
}

def parse_face_value(denomination_str: str) -> float:
    parts = denomination_str.strip().split()
    if not parts:
        return 0.0
    try:
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]) / int(parts[1])
        return float(parts[0].replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return 0.0

def coin_to_krw(class_str_id: str) -> float:
    info = cat_to_name.get(class_str_id, "")
    if not info:
        return 0.0
    parts = [p.strip() for p in info.split(",")]
    if len(parts) < 2:
        return 0.0
    rate = EXCHANGE_RATES.get(parts[1].lower(), 0.0)
    return parse_face_value(parts[0]) * rate

# ─── 동전 검출 (원형도 필터) ──────────────────────────────────────────────────
def detect_coin_circles(th, min_radius=25, circularity_thresh=0.6):
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
        if radius > min_radius and circularity >= circularity_thresh:
            circles.append(((cx, cy), radius))
    circles.sort(key=lambda x: (x[0][0], x[0][1]))
    return circles

# ─── 메인 추론 함수 ───────────────────────────────────────────────────────────
def predict_coins(image_array, min_radius, circularity_thresh, conf_thresh):
    src = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = cv2.bitwise_not(th)
    # 정방향 + 역방향 이진화를 합쳐서 검정/흰 배경 모두 대응
    th_combined = cv2.bitwise_or(th, th_inv)
    th_open = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    th_inv_open = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    circles_fwd = detect_coin_circles(th_open, min_radius, circularity_thresh)
    circles_inv = detect_coin_circles(th_inv_open, min_radius, circularity_thresh)

    # 두 방향에서 검출된 원을 합치고 중복 제거 (중심 거리 기준)
    all_circles = list(circles_fwd)
    for c2 in circles_inv:
        cx2, cy2 = c2[0]
        is_dup = any(
            ((cx2 - c1[0][0])**2 + (cy2 - c1[0][1])**2) < (c1[1] * 0.7)**2
            for c1 in all_circles
        )
        if not is_dup:
            all_circles.append(c2)

    circles = sorted(all_circles, key=lambda x: (x[0][0], x[0][1]))

    if not circles:
        return image_array, [], 0.0, 0

    coin_imgs = []
    for center, radius in circles:
        r = max(2, int(radius * 2.5))
        cen = (r // 2, r // 2)
        mask = np.zeros((r, r, 3), np.uint8)
        cv2.circle(mask, cen, radius, (255, 255, 255), cv2.FILLED)
        coin = cv2.getRectSubPix(src, (r, r), center)
        coin = cv2.bitwise_and(coin, mask)
        coin_imgs.append(coin)

    preds = model.predict(coin_imgs, imgsz=224, verbose=False)
    result_img = src.copy()
    total_krw = 0.0
    unknown_count = 0
    coin_results = []

    for pred, (center, radius) in zip(preds, circles):
        cls_id = int(pred.probs.top1)
        class_str_id = model.names[cls_id]
        readable_name = cat_to_name.get(class_str_id, f"Class {class_str_id}")
        conf = float(pred.probs.top1conf)
        cx, cy = int(center[0]), int(center[1])

        if conf < conf_thresh:
            short_name = "?"
            krw_val = 0.0
            unknown_count += 1
            color = (0, 0, 255)
        else:
            short_name = readable_name.split(",")[0] if "," in readable_name else readable_name
            krw_val = coin_to_krw(class_str_id)
            total_krw += krw_val
            color = (0, 255, 0)
            coin_results.append({
                "name": readable_name,
                "short": short_name,
                "conf": conf,
                "krw": krw_val,
            })

        cv2.circle(result_img, (cx, cy), radius, color, 3)
        label = f"{short_name} ({conf:.0%})"
        cv2.putText(result_img, label, (cx - radius, cy - radius - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    return result_img_rgb, coin_results, total_krw, unknown_count


# ─── Streamlit UI ─────────────────────────────────────────────────────────────
st.title("🪙 AI 글로벌 동전 분류기 (YOLOv8)")
st.markdown("동전 사진을 업로드하면 AI가 전세계 **211종** 통화를 자동 분류하고 **한화 기준 총액**을 계산합니다.")
st.divider()

# ─── 사이드바: 감지 파라미터 슬라이더 ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 감지 파라미터 조절")
    st.markdown("인식이 잘 안 될 때 아래 값을 조정해보세요.")
    st.divider()

    min_radius = st.slider(
        "최소 반지름 (px)",
        min_value=10, max_value=80, value=25, step=5,
        help="작은 동전 감지가 안 될 때 값을 낮추세요."
    )
    circularity = st.slider(
        "원형도 임계치",
        min_value=0.3, max_value=0.95, value=0.60, step=0.05,
        help="노이즈가 동전으로 잡힐 때 값을 높이고, 동전이 안 잡힐 때 낮추세요."
    )
    conf_thresh = st.slider(
        "분류 신뢰도 임계치",
        min_value=0.1, max_value=0.9, value=0.30, step=0.05,
        help="빨간 원이 너무 많을 때 높이고, 초록 원이 너무 없을 때 낮추세요."
    )
    st.divider()
    st.caption("🟢 초록 원 = 정상 인식\n🔴 빨간 원 = 신뢰도 미달")

# 모델 로드 확인
if model is None:
    st.error("❗ 'best.pt' 모델 파일을 찾을 수 없습니다.")
    st.stop()

col_left, col_right = st.columns(2)

with col_left:
    uploaded = st.file_uploader("동전 이미지 업로드", type=["jpg", "jpeg", "png", "webp"])

    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="업로드된 이미지", use_container_width=True)

        run_btn = st.button("🔍 분류 시작", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("AI가 동전을 분석 중입니다..."):
                result_img, coin_results, total_krw, unknown_count = predict_coins(
                    img_rgb, min_radius, circularity, conf_thresh
                )

            st.markdown(
                f"""
                <div style="
                    background:#eef4ff; border:2px solid #b3d0f5; border-radius:12px;
                    padding:20px; text-align:center; margin-top:12px;
                ">
                    <div style="font-size:0.95em; color:#555;">💰 한화 기준 총액</div>
                    <div style="font-size:2.4em; font-weight:800; color:#1a6bcc;">
                        {total_krw:,.0f} 원
                    </div>
                    <div style="font-size:0.8em; color:#888;">※ 2026년 2월 기준 환율 근사치</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with col_right:
    if uploaded and 'result_img' in dir():
        st.image(result_img, caption="분석 결과 이미지 (초록=인식, 빨강=미달)", use_container_width=True)

        st.markdown("#### 🪙 분류 결과 상세")
        if coin_results:
            rows = []
            counts = Counter(r["name"] for r in coin_results)
            seen = set()
            for r in coin_results:
                if r["name"] in seen:
                    continue
                seen.add(r["name"])
                parts = r["name"].split(",")
                coin_label = parts[0].strip()
                country = parts[2].strip().capitalize() if len(parts) > 2 else ""
                rows.append({
                    "동전": f"{coin_label} [{country}]",
                    "수량": counts[r["name"]],
                    "단가(원)": f"≈ {r['krw']:,.0f}원",
                    "소계(원)": f"≈ {r['krw'] * counts[r['name']]:,.0f}원",
                })
            st.table(rows)

        if unknown_count > 0:
            st.warning(f"⚠️ 신뢰도 미달 동전: {unknown_count}개 (총액 미포함)")
