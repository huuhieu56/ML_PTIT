import os
import json
import joblib
import pandas as pd
import streamlit as st
from src.preprocess import clean_text

st.set_page_config(page_title="Review Sentiment & Rating", page_icon="🛍️", layout="wide")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

SENTIMENT_MAP = {0: "🔴 Tiêu cực", 1: "🟡 Trung tính", 2: "🟢 Tích cực"}
MODELS = ["ensemble_voting", "linear_svm", "logistic_regression", "lightgbm", "naive_bayes"]

st.title("🛍️ Phân Tích Cảm Xúc & Xếp Hạng Đánh Giá Sản Phẩm")

tab_pred, tab_bench, tab_err = st.tabs(["🎯 Dự Đoán", "📊 Benchmark", "🔍 Phân Tích Lỗi"])

with tab_pred:
    col1, col2 = st.columns([1, 2])
    with col1:
        lang = "vi" if st.selectbox("Ngôn ngữ:", ["Tiếng Việt", "English"]) == "Tiếng Việt" else "en"
        model_name = st.selectbox("Mô hình:", MODELS)
        sample = st.radio("Mẫu thử:", [
            "Điện thoại dùng rất mượt, pin trâu chụp ảnh đẹp.",
            "Giao hàng quá chậm, đóng gói rách nát, liệt cảm ứng.",
            "Máy tạm được trong tầm giá, pin bình thường.",
            "Shop phục vụ 'tuyệt vời' ghê, nhắn tin không trả lời."
        ] if lang == "vi" else [
            "Great product, works perfectly and fast shipping.",
            "Arrived broken, terrible customer service.",
            "Average quality, okay for the price.",
            "Oh what a wonderful piece of junk."
        ])

    with col2:
        text = st.text_area("Nội dung đánh giá:", value=sample, height=100)
        if st.button("Dự Đoán", type="primary"):
            cleaned = clean_text(text, is_vietnamese=(lang == "vi"))
            st.info(f"Văn bản xử lý: `{cleaned}`")

            sent_pipe = joblib.load(os.path.join(MODELS_DIR, f"ml_{lang}_sentiment_{model_name}.joblib"))
            rate_pipe = joblib.load(os.path.join(MODELS_DIR, f"ml_{lang}_rating_{model_name}.joblib"))

            pred_sent = sent_pipe.predict([cleaned])[0]
            pred_rate = rate_pipe.predict([cleaned])[0]

            c1, c2 = st.columns(2)
            with c1:
                st.subheader(f"Cảm xúc: {SENTIMENT_MAP.get(pred_sent, '')}")
                probs = sent_pipe.predict_proba([cleaned])[0]
                st.bar_chart(pd.DataFrame({"Xác suất": probs}, index=["Tiêu cực", "Trung tính", "Tích cực"]))

            with c2:
                st.subheader(f"Xếp hạng: {'⭐' * int(pred_rate)} ({pred_rate}/5★)")
                probs_r = rate_pipe.predict_proba([cleaned])[0]
                st.bar_chart(pd.DataFrame({"Xác suất": probs_r}, index=["1★", "2★", "3★", "4★", "5★"]))

with tab_bench:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Tiếng Việt (Sentiment & Rating)")
        st.dataframe(pd.read_csv(os.path.join(REPORTS_DIR, "benchmark_vi_sentiment.csv")), use_container_width=True)
        st.dataframe(pd.read_csv(os.path.join(REPORTS_DIR, "benchmark_vi_rating.csv")), use_container_width=True)
    with c2:
        st.markdown("### English (Sentiment & Rating)")
        st.dataframe(pd.read_csv(os.path.join(REPORTS_DIR, "benchmark_en_sentiment.csv")), use_container_width=True)
        st.dataframe(pd.read_csv(os.path.join(REPORTS_DIR, "benchmark_en_rating.csv")), use_container_width=True)

with tab_err:
    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(FIG_DIR, "cm_vi_sentiment_svm.png"), caption="Confusion Matrix VI")
    with c2:
        st.image(os.path.join(FIG_DIR, "cm_en_sentiment_svm.png"), caption="Confusion Matrix EN")

    st.markdown("### Mẫu câu mô hình đoán sai (Failure cases)")
    with open(os.path.join(REPORTS_DIR, "vi_error_cases.json")) as f:
        cases = json.load(f)["cases"]
    st.dataframe(pd.DataFrame(cases), use_container_width=True)
