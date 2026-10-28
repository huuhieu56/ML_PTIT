from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

def get_pipelines():
    """Return dictionary of classical ML pipelines."""
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42, dual=False))
    nb = MultinomialNB(alpha=0.5)
    lgb = LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1)
    ens = VotingClassifier([("lr", lr), ("svm", svm), ("nb", nb)], voting="soft")

    models = {
        "naive_bayes": nb,
        "logistic_regression": lr,
        "linear_svm": svm,
        "lightgbm": lgb,
        "ensemble_voting": ens
    }
    return {name: Pipeline([("tfidf", vec), ("clf", clf)]) for name, clf in models.items()}
