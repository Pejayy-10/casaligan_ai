import os
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump
from dateutil.parser import parse as dt_parse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score
)

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Reference-grounded taxonomy (skills/tasks categories) ----
# (Grounding references for your documentation: TESDA Housekeeping NC II, ILO Domestic Work, O*NET tasks, IRR RA10361)
SKILLS = [
    "cleaning", "deep cleaning", "laundry", "ironing", "cooking",
    "baby care", "elder care", "pet care", "gardening",
    "house organization"
]
LANGUAGES = ["english", "filipino", "chavacano", "bisaya"]
CITIES = [
    "Zamboanga City", "Tetuan", "Guiwan", "Putik", "Canelar",
    "Divisoria", "Ayala", "Sta. Maria", "Sangali"
]
GENDERS = ["female", "male"]

# ---- Intent training samples ----
# Keep this small but diverse; add more examples anytime.
INTENT_CLASSES = ["RECOMMEND", "SUPPORT", "JOB_POST", "SMALLTALK", "OFF_TOPIC"]

def build_intent_samples() -> pd.DataFrame:
    samples = []

    # RECOMMEND
    rec = [
        "recommend a housekeeper",
        "find me a helper for cleaning",
        "looking for a female housekeeper",
        "need baby care from jan 12 to jan 14",
        "chavacano housekeeper near tetuan",
        "hire a yaya who can cook",
        "available on 2026-01-12 cleaning budget 650",
        "suggest someone for laundry and ironing",
        "i want a kasambahay near guiwan",
        "show me housekeepers that are available tomorrow",
    ]
    for t in rec:
        samples.append((t, "RECOMMEND"))

    # SUPPORT
    sup = [
        "i can't login",
        "password not working",
        "how to reset my password",
        "payment failed",
        "my screen is stuck",
        "the app is not loading",
        "why can't i open the profile",
        "i got an error message",
        "how do i contact support",
    ]
    for t in sup:
        samples.append((t, "SUPPORT"))

    # JOB_POST
    jp = [
        "how do i post a job",
        "help me create a job post",
        "i want to post housekeeping work",
        "how can housekeepers apply to my post",
        "how to set my job schedule and budget",
    ]
    for t in jp:
        samples.append((t, "JOB_POST"))

    # SMALLTALK
    st = [
        "hello",
        "hi casaligan",
        "good morning",
        "who are you",
        "what can you do",
        "thanks",
        "thank you so much",
        "nice",
    ]
    for t in st:
        samples.append((t, "SMALLTALK"))

    # OFF_TOPIC
    ot = [
        "what is the capital of france",
        "solve this math problem",
        "write me a poem",
        "who is the president of the philippines",
        "give me minecraft mod suggestions",
        "explain calculus",
    ]
    for t in ot:
        samples.append((t, "OFF_TOPIC"))

    df = pd.DataFrame(samples, columns=["text", "label"])
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df

# ---- Availability helpers ----
def random_blocked_ranges() -> str:
    # 0 to 3 blocked ranges
    k = random.randint(0, 3)
    ranges = []
    base = date.today()
    for _ in range(k):
        start = base + timedelta(days=random.randint(1, 60))
        end = start + timedelta(days=random.randint(0, 5))
        ranges.append(f"{start.isoformat()}:{end.isoformat()}")
    return "|".join(ranges)

def parse_blocked(blocked_str: str) -> List[Tuple[date, date]]:
    if not isinstance(blocked_str, str) or blocked_str.strip() == "":
        return []
    out = []
    for part in blocked_str.split("|"):
        if ":" not in part:
            continue
        s, e = part.split(":", 1)
        out.append((dt_parse(s).date(), dt_parse(e).date()))
    return out

def overlap(a1: date, a2: date, b1: date, b2: date) -> bool:
    return not (a2 < b1 or b2 < a1)

def is_available(blocked: List[Tuple[date, date]], rs: date, re_: date) -> bool:
    for bs, be in blocked:
        if overlap(rs, re_, bs, be):
            return False
    return True

# ---- Reco feature engineering (must match app.py) ----
RECO_FEATURES = [
    "pref_min_age", "pref_budget", "pref_skill_count",
    "hk_age", "hk_experience_years", "hk_price",
    "skill_overlap", "age_ok", "gender_ok", "city_match", "budget_ok",
    "lang_overlap", "language_ok",
    "availability_ok", "overlap_days"
]

def build_features(pref: Dict[str, any], hk_row: pd.Series) -> Dict[str, float]:
    hk_skills = set(str(hk_row["skills"]).split("|"))
    hk_langs = set(str(hk_row["languages"]).split("|"))

    skill_overlap = len(set(pref["skills"]) & hk_skills) if pref["skills"] else 0

    age_ok = 1 if (pref["min_age"] is None or int(hk_row["age"]) >= int(pref["min_age"])) else 0
    gender_ok = 1 if (pref["gender"] is None or str(hk_row["gender"]).lower() == pref["gender"]) else 0
    city_match = 1 if (pref["city"] is None or str(hk_row["city"]) == pref["city"]) else 0

    budget_ok = 1
    if pref["budget"] is not None:
        budget_ok = 1 if int(hk_row["price"]) <= int(pref["budget"]) else 0

    lang_overlap = 0
    language_ok = 1
    if pref["language"] is not None:
        lang_overlap = 1 if pref["language"] in hk_langs else 0
        language_ok = lang_overlap

    availability_ok = 1
    overlap_days = 0
    if pref["date_start"]:
        rs = dt_parse(pref["date_start"]).date()
        re_ = dt_parse(pref["date_end"]).date() if pref["date_end"] else rs
        blocked = parse_blocked(str(hk_row["blocked_ranges"]))
        availability_ok = 1 if is_available(blocked, rs, re_) else 0
        if availability_ok == 0:
            overlap_days = 2

    return {
        "pref_min_age": float(pref["min_age"] or 0),
        "pref_budget": float(pref["budget"] or 0),
        "pref_skill_count": float(len(pref["skills"])),

        "hk_age": float(hk_row["age"]),
        "hk_experience_years": float(hk_row["experience_years"]),
        "hk_price": float(hk_row["price"]),

        "skill_overlap": float(skill_overlap),
        "age_ok": float(age_ok),
        "gender_ok": float(gender_ok),
        "city_match": float(city_match),
        "budget_ok": float(budget_ok),

        "lang_overlap": float(lang_overlap),
        "language_ok": float(language_ok),

        "availability_ok": float(availability_ok),
        "overlap_days": float(overlap_days),
    }

def random_pref() -> Dict[str, any]:
    # Generate realistic-ish user preferences
    skills = random.sample(SKILLS, k=random.randint(1, 3))
    language = random.choice(LANGUAGES + [None, None])  # sometimes none
    city = random.choice(CITIES + [None, None])
    gender = random.choice(GENDERS + [None, None, None])
    min_age = random.choice([None, None, 18, 20, 22, 25])
    budget = random.choice([None, None, 200, 350, 500, 650, 800])

    # sometimes ask for date range
    if random.random() < 0.55:
        start = date.today() + timedelta(days=random.randint(1, 30))
        end = start + timedelta(days=random.randint(0, 5))
        ds, de = start.isoformat(), end.isoformat()
    else:
        ds, de = None, None

    return {
        "skills": skills,
        "language": language,
        "city": city,
        "gender": gender,
        "min_age": min_age,
        "budget": budget,
        "date_start": ds,
        "date_end": de,
    }

def label_match(pref: Dict[str, any], hk: pd.Series) -> int:
    feats = build_features(pref, hk)
    # Label rule: must overlap skills >= 1
    if feats["skill_overlap"] < 1:
        return 0
    # If user asked language, require it
    if pref["language"] is not None and feats["language_ok"] == 0:
        return 0
    # If user asked dates, require availability
    if pref["date_start"] is not None and feats["availability_ok"] == 0:
        return 0
    # If budget asked, require budget_ok
    if pref["budget"] is not None and feats["budget_ok"] == 0:
        return 0
    # Optional constraints: gender, age, city (soft)
    # We treat them as helpful but not strict in labels (to avoid too-hard labels).
    return 1

def train_intent_model() -> Dict[str, any]:
    df = build_intent_samples()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=SEED, stratify=df["label"]
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, y_train)

    pred = clf.predict(Xte)

    acc = accuracy_score(y_test, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=INTENT_CLASSES).tolist()

    print("\n==== INTENT MODEL EVALUATION ====")
    print("Accuracy:", round(acc, 4))
    print("Precision (weighted):", round(pr, 4))
    print("Recall (weighted):", round(rc, 4))
    print("F1 (weighted):", round(f1, 4))
    print("\nClassification Report:\n", classification_report(y_test, pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred) labels=", INTENT_CLASSES)
    print(np.array(cm))

    dump(vec, os.path.join(ART, "intent_vectorizer.joblib"))
    dump(clf, os.path.join(ART, "intent_model.joblib"))

    return {
        "intent": {
            "accuracy": float(acc),
            "precision_weighted": float(pr),
            "recall_weighted": float(rc),
            "f1_weighted": float(f1),
            "labels": INTENT_CLASSES,
            "confusion_matrix": cm
        }
    }

def train_reco_model(hk_df: pd.DataFrame) -> Dict[str, any]:
    # Generate training pairs (pref, hk) and labels
    rows = []
    for _ in range(1400):  # increase if you want
        pref = random_pref()
        hk = hk_df.sample(1, random_state=random.randint(1, 999999)).iloc[0]
        y = label_match(pref, hk)
        feats = build_features(pref, hk)
        rows.append((feats, y))

    feat_df = pd.DataFrame([r[0] for r in rows])
    y = np.array([r[1] for r in rows], dtype=int)

    # Balance a bit (optional)
    # Keep all positives, sample negatives
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) > 0:
        keep_neg = np.random.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 2), replace=False)
        keep_idx = np.concatenate([pos_idx, keep_neg])
        feat_df = feat_df.iloc[keep_idx].reset_index(drop=True)
        y = y[keep_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        feat_df[RECO_FEATURES], y, test_size=0.25, random_state=SEED, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else float("nan")

    print("\n==== RECOMMENDATION MODEL EVALUATION ====")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(pr, 4))
    print("Recall:", round(rc, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4) if not np.isnan(auc) else "N/A")

    dump(model, os.path.join(ART, "reco_model.joblib"))
    dump(RECO_FEATURES, os.path.join(ART, "reco_features.joblib"))

    return {
        "recommender": {
            "accuracy": float(acc),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "roc_auc": float(auc) if not np.isnan(auc) else None,
            "n_samples": int(len(y))
        }
    }

def main():
    hk_path = os.path.join(ART, "housekeepers.csv")
    if not os.path.exists(hk_path):
        raise FileNotFoundError(
            f"Missing {hk_path}. Put your housekeepers.csv inside artifacts/ first."
        )

    hk = pd.read_csv(hk_path)

    # Ensure blocked_ranges exists (if not, create)
    if "blocked_ranges" not in hk.columns:
        hk["blocked_ranges"] = [random_blocked_ranges() for _ in range(len(hk))]

    metrics = {}
    metrics.update(train_intent_model())
    metrics.update(train_reco_model(hk))

    out = os.path.join(ART, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved metrics to:", out)
    print("Saved artifacts to:", ART)

if __name__ == "__main__":
    main()
