import os
import json
import time
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from src.ml_models import get_pipelines

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")

def train_and_eval(lang: str, task: str):
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"{lang}_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, f"{lang}_test.csv"))
    label_col = "sentiment" if task == "sentiment" else "rating"

    X_train, y_train = train_df["clean_text"].astype(str), train_df[label_col].astype(int)
    X_test, y_test = test_df["clean_text"].astype(str), test_df[label_col].astype(int)

    results = {}
    print(f"\n=== {lang.upper()} | {task.upper()} ({len(X_train)} train, {len(X_test)} test) ===")

    for name, pipeline in get_pipelines().items():
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        duration = round(time.time() - t0, 2)
        preds = pipeline.predict(X_test)

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y_test, preds, average="weighted")), 4),
            "train_time_sec": duration,
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
        if task == "rating":
            metrics["mae"] = round(float(mean_absolute_error(y_test, preds)), 4)

        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"ml_{lang}_{task}_{name}.joblib"))
        print(f"[{name:<18}] Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Time: {duration}s")
        results[name] = metrics

    return results

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    all_results = {
        "vi_sentiment": train_and_eval("vi", "sentiment"),
        "vi_rating": train_and_eval("vi", "rating"),
        "en_sentiment": train_and_eval("en", "sentiment"),
        "en_rating": train_and_eval("en", "rating")
    }
    with open(os.path.join(REPORTS_DIR, "ml_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved all models and metrics.")

if __name__ == "__main__":
    main()
