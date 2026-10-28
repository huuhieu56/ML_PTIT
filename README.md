# Sentiment & Rating Prediction

Hệ thống phân tích cảm xúc (3 lớp) và dự đoán xếp hạng (1-5★) dùng Machine Learning truyền thống (Naive Bayes, Logistic Regression, Linear SVM, LightGBM, Soft Voting Ensemble) trên 2 bộ dữ liệu (Tiếng Việt & Tiếng Anh).

## Cài đặt
```bash
pip install scikit-learn lightgbm pandas numpy matplotlib seaborn streamlit underthesea
```

## Sử dụng
```bash
python train_ml.py        # 1. Huấn luyện mô hình
python evaluate_all.py    # 2. Xuất bảng chỉ số & biểu đồ
streamlit run app.py      # 3. Mở web demo
```
# ML_PTIT
