import os
import pandas as pd
from datasets import load_dataset
from src.preprocess import clean_text, rating_to_sentiment

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def prepare_vietnamese_dataset(force_download: bool = False, segment_words: bool = True):
    """Fetch, clean and cache Vietnamese Product Reviews dataset."""
    train_path = os.path.join(DATA_DIR, "vi_train.csv")
    test_path = os.path.join(DATA_DIR, "vi_test.csv")

    if not force_download and os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    print("[DataLoader] Downloading Vietnamese dataset (anotherpolarbear/vietnamese-sentiment-analysis)...")
    ds = load_dataset("anotherpolarbear/vietnamese-sentiment-analysis")

    def process_split(split_data):
        df = pd.DataFrame(split_data)
        df = df.rename(columns={"comment": "raw_text", "label": "rating"})
        df["rating"] = df["rating"].astype(int)
        df["sentiment"] = df["rating"].apply(rating_to_sentiment)
        print(f"[DataLoader] Cleaning {len(df)} Vietnamese samples...")
        df["clean_text"] = df["raw_text"].apply(lambda t: clean_text(t, is_vietnamese=True, segment_words=segment_words))
        df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
        return df

    train_df = process_split(ds["train"])
    test_df = process_split(ds["test"])

    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[DataLoader] Saved Vietnamese dataset -> {train_path} ({len(train_df)}), {test_path} ({len(test_df)})")
    return train_df, test_df

def prepare_english_dataset(force_download: bool = False, n_train: int = 10000, n_test: int = 2000):
    """Fetch, clean and cache Amazon English Reviews dataset."""
    train_path = os.path.join(DATA_DIR, "en_train.csv")
    test_path = os.path.join(DATA_DIR, "en_test.csv")

    if not force_download and os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    print("[DataLoader] Downloading Amazon English dataset (SetFit/amazon_reviews_multi_en)...")
    ds = load_dataset("SetFit/amazon_reviews_multi_en")

    def process_split(split_data, sample_n=None):
        df = pd.DataFrame(split_data)
        if sample_n and len(df) > sample_n:
            df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        df = df.rename(columns={"text": "raw_text"})
        df["rating"] = df["label"].astype(int) + 1  # 0..4 -> 1..5
        df["sentiment"] = df["rating"].apply(rating_to_sentiment)
        print(f"[DataLoader] Cleaning {len(df)} English samples...")
        df["clean_text"] = df["raw_text"].apply(lambda t: clean_text(t, is_vietnamese=False, segment_words=False))
        df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
        return df[["raw_text", "clean_text", "rating", "sentiment"]]

    train_df = process_split(ds["train"], sample_n=n_train)
    test_df = process_split(ds["test"], sample_n=n_test)

    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[DataLoader] Saved English dataset -> {train_path} ({len(train_df)}), {test_path} ({len(test_df)})")
    return train_df, test_df

if __name__ == "__main__":
    prepare_vietnamese_dataset()
    prepare_english_dataset()
