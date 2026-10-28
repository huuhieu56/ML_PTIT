import os
import json
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from src.error_analysis import plot_confusion_matrix, extract_error_cases, SENTIMENT_LABELS, RATING_LABELS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

def main():
    with open(os.path.join(REPORTS_DIR, "ml_results.json"), "r") as f:
        data = json.load(f)

    # 1. Export benchmark tables
    for key, task_name in [("vi_sentiment", "VI Sentiment"), ("vi_rating", "VI Rating"),
                           ("en_sentiment", "EN Sentiment"), ("en_rating", "EN Rating")]:
        rows = [{"Model": k, **v} for k, v in data[key].items()]
        df = pd.DataFrame(rows).drop(columns=["confusion_matrix"])
        df.to_csv(os.path.join(REPORTS_DIR, f"benchmark_{key}.csv"), index=False)

    # 2. Confusion Matrices & Error Mining
    vi_test = pd.read_csv(os.path.join(DATA_DIR, "vi_test.csv"))
    svm_vi = joblib.load(os.path.join(MODELS_DIR, "ml_vi_sentiment_linear_svm.joblib"))
    vi_preds = svm_vi.predict(vi_test["clean_text"].astype(str))

    plot_confusion_matrix(
        confusion_matrix(vi_test["sentiment"], vi_preds),
        SENTIMENT_LABELS,
        "Confusion Matrix: SVM on VI Sentiment",
        os.path.join(FIG_DIR, "cm_vi_sentiment_svm.png")
    )

    vi_errs = extract_error_cases(vi_test["clean_text"], vi_test["sentiment"], vi_preds, SENTIMENT_LABELS)
    with open(os.path.join(REPORTS_DIR, "vi_error_cases.json"), "w", encoding="utf-8") as f:
        json.dump(vi_errs, f, indent=2, ensure_ascii=False)

    en_test = pd.read_csv(os.path.join(DATA_DIR, "en_test.csv"))
    svm_en = joblib.load(os.path.join(MODELS_DIR, "ml_en_sentiment_linear_svm.joblib"))
    en_preds = svm_en.predict(en_test["clean_text"].astype(str))

    plot_confusion_matrix(
        confusion_matrix(en_test["sentiment"], en_preds),
        SENTIMENT_LABELS,
        "Confusion Matrix: SVM on EN Sentiment",
        os.path.join(FIG_DIR, "cm_en_sentiment_svm.png")
    )

    en_errs = extract_error_cases(en_test["clean_text"], en_test["sentiment"], en_preds, SENTIMENT_LABELS)
    with open(os.path.join(REPORTS_DIR, "en_error_cases.json"), "w", encoding="utf-8") as f:
        json.dump(en_errs, f, indent=2, ensure_ascii=False)

    print("Evaluations and error analysis saved.")

if __name__ == "__main__":
    main()
