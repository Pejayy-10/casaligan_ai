import os
import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from joblib import dump
from dateutil.parser import parse as dt_parse

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------
# Domain setup
# ----------------------------
SKILLS = [
    "cleaning", "laundry", "ironing", "cooking", "baby care", "elder care",
    "pet care", "gardening", "house organization", "deep cleaning"
]

CITIES = [
    "Zamboanga City", "Tetuan", "Guiwan", "Putik", "Canelar",
    "Divisoria", "Ayala", "Sta. Maria", "Sangali"
]

LANGUAGES = ["english", "filipino", "chavacano", "bisaya"]
GENDERS = ["female", "male"]
PACKAGE_TYPES = ["hourly", "daily", "weekly"]


def weighted_choice(items, weights):
    r = random.random() * sum(weights)
    upto = 0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]


@dataclass
class Housekeeper:
    housekeeper_id: str
    name: str
    age: int
    gender: str
    city: str
    skills: List[str]
    experience_years: int
    languages: List[str]
    package_type: str
    price: int
    blocked_ranges: List[Tuple[str, str]]


def rand_name():
    first = ["Aira", "Joan", "Mark", "Riza", "Kyla", "Nina", "Alex", "Carlo", "Mae", "Ruth", "Jessa", "Liza"]
    last = ["Santos", "Reyes", "Garcia", "Dela Cruz", "Flores", "Torres", "Villanueva", "Gonzales"]
    return f"{random.choice(first)} {random.choice(last)}"


def random_blocked_ranges(start_day: date, days_span=150, max_ranges=4):
    ranges = []
    for _ in range(random.randint(0, max_ranges)):
        s = start_day + timedelta(days=random.randint(0, days_span - 1))
        e = s + timedelta(days=random.randint(1, 4))
        ranges.append((s.isoformat(), e.isoformat()))
    return ranges


def generate_housekeepers(n=300, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    today = date.today()

    keepers = []
    for i in range(n):
        hid = f"HK-{i+1:04d}"
        age = random.randint(18, 55)
        gender = random.choice(GENDERS)
        city = random.choice(CITIES)

        skills = random.sample(SKILLS, k=random.randint(2, 5))
        exp = max(0, int(np.random.normal(loc=3.5, scale=2.0)))

        # Weighted for Zambo context: chavacano + filipino often
        lang1 = weighted_choice(LANGUAGES, weights=[0.20, 0.35, 0.35, 0.10])
        lang2 = weighted_choice(LANGUAGES, weights=[0.25, 0.35, 0.25, 0.15])
        languages = sorted(list(set([lang1, lang2])))[:2]

        package_type = random.choice(PACKAGE_TYPES)
        base = {"hourly": 90, "daily": 550, "weekly": 2800}[package_type]
        price = int(max(base * 0.7, base + np.random.normal(0, base * 0.18)))

        blocked = random_blocked_ranges(today, 150, 4)

        keepers.append(Housekeeper(
            housekeeper_id=hid,
            name=rand_name(),
            age=age,
            gender=gender,
            city=city,
            skills=skills,
            experience_years=exp,
            languages=languages,
            package_type=package_type,
            price=price,
            blocked_ranges=blocked
        ))
    return keepers


def ranges_overlap(a_start: date, a_end: date, b_start: date, b_end: date):
    return not (a_end < b_start or b_end < a_start)


def is_available(blocked_ranges: List[Tuple[str, str]], req_start: date, req_end: date) -> bool:
    for s, e in blocked_ranges:
        bs = dt_parse(s).date()
        be = dt_parse(e).date()
        if ranges_overlap(req_start, req_end, bs, be):
            return False
    return True


def make_query_text(pref: Dict[str, Any]) -> str:
    parts = ["Find me a housekeeper"]
    if pref.get("gender"):
        parts.append(pref["gender"])
    if pref.get("min_age"):
        parts.append(f"above {pref['min_age']}")
    if pref.get("language"):
        parts.append(pref["language"])
    if pref.get("city"):
        parts.append(f"near {pref['city']}")
    if pref.get("skills"):
        parts.append("skills: " + ", ".join(pref["skills"]))
    if pref.get("budget"):
        parts.append(f"budget {pref['budget']}")
    if pref.get("date_start"):
        if pref.get("date_end") and pref["date_end"] != pref["date_start"]:
            parts.append(f"from {pref['date_start']} to {pref['date_end']}")
        else:
            parts.append(f"on {pref['date_start']}")
    return " ".join(parts)


def generate_training_pairs(keepers: List[Housekeeper], n_queries=1400, pairs_per_query=30, seed=42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    today = date.today()

    rows = []
    for _ in range(n_queries):
        pref = {
            "gender": random.choice([None, "female", "male"]),
            "min_age": random.choice([None, 20, 22, 25, 30]),
            "language": random.choice([None, "english", "filipino", "chavacano", "bisaya"]),
            "city": random.choice([None] + CITIES),
            "skills": random.sample(SKILLS, k=random.randint(1, 3)),
            "budget": random.choice([None, 120, 150, 200, 550, 650, 2800, 3200]),
        }

        if random.random() < 0.75:
            start = today + timedelta(days=random.randint(0, 70))
            end = start + timedelta(days=random.randint(0, 3))
            pref["date_start"] = start.isoformat()
            pref["date_end"] = end.isoformat()
        else:
            pref["date_start"] = None
            pref["date_end"] = None

        query_text = make_query_text(pref)

        candidates = random.sample(keepers, k=min(pairs_per_query, len(keepers)))
        for hk in candidates:
            skill_overlap = len(set(pref["skills"]) & set(hk.skills))

            age_ok = 1 if (pref["min_age"] is None or hk.age >= pref["min_age"]) else 0
            gender_ok = 1 if (pref["gender"] is None or hk.gender == pref["gender"]) else 0
            city_match = 1 if (pref["city"] is None or hk.city == pref["city"]) else 0

            budget_ok = 1
            if pref["budget"] is not None:
                budget_ok = 1 if hk.price <= pref["budget"] else 0

            lang_overlap = 0
            language_ok = 1
            if pref["language"] is not None:
                lang_overlap = 1 if pref["language"] in [x.lower() for x in hk.languages] else 0
                language_ok = lang_overlap

            availability_ok = 1
            overlap_days = 0
            if pref["date_start"]:
                rs = dt_parse(pref["date_start"]).date()
                re_ = dt_parse(pref["date_end"]).date()
                availability_ok = 1 if is_available(hk.blocked_ranges, rs, re_) else 0
                if availability_ok == 0:
                    overlap_days = random.randint(1, 3)

            label = 1 if (
                skill_overlap >= 1
                and age_ok == 1
                and gender_ok == 1
                and language_ok == 1
                and availability_ok == 1
                and budget_ok == 1
            ) else 0

            if random.random() < 0.03:
                label = 1 - label

            rows.append({
                "query_text": query_text,

                "pref_min_age": float(pref["min_age"] or 0),
                "pref_budget": float(pref["budget"] or 0),
                "pref_skill_count": float(len(pref["skills"])),

                "hk_age": float(hk.age),
                "hk_experience_years": float(hk.experience_years),
                "hk_price": float(hk.price),

                "skill_overlap": float(skill_overlap),
                "age_ok": float(age_ok),
                "gender_ok": float(gender_ok),
                "city_match": float(city_match),
                "budget_ok": float(budget_ok),

                "lang_overlap": float(lang_overlap),
                "language_ok": float(language_ok),

                "availability_ok": float(availability_ok),
                "overlap_days": float(overlap_days),

                "label_match": int(label),
            })

    return pd.DataFrame(rows)


def generate_intent_dataset(seed=42) -> pd.DataFrame:
    random.seed(seed)
    today = date.today()

    def iso(d): return d.isoformat()

    recommend = [
        "can you suggest me a housekeeper that is female and speaks chavacano",
        "can you find me a maid who speaks bisaya",
        "recommend a helper near tetuan budget 650",
        "looking for a female housekeeper 22+ baby care",
        "need cooking and cleaning available on " + iso(today + timedelta(days=5)),
        "find me a housekeeper who can do laundry under 200",
        "i want to hire a yaya who speaks chavacano",
        "match me with a housekeeper for deep cleaning",
    ]

    job_post = [
        "how do i post a job?",
        "help me create a job posting",
        "what should i write in my job post?",
        "how do i set my budget in job post?",
    ]

    support = [
        "i can't login",
        "how do i reset my password",
        "my payment failed",
        "how do refunds work",
        "where can i see my applications",
        "my account got locked",
    ]

    smalltalk = [
        "hello",
        "good morning",
        "thanks",
        "who are you?",
        "what can you do?",
        "nice!",
    ]

    rows = []
    for _ in range(520):
        rows.append((random.choice(recommend), "RECOMMEND"))
    for _ in range(220):
        rows.append((random.choice(job_post), "JOB_POST"))
    for _ in range(260):
        rows.append((random.choice(support), "SUPPORT"))
    for _ in range(260):
        rows.append((random.choice(smalltalk), "SMALLTALK"))

    df = pd.DataFrame(rows, columns=["text", "intent"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    os.makedirs("artifacts", exist_ok=True)

    keepers = generate_housekeepers(n=300, seed=42)

    # Save housekeepers
    hk_rows = []
    for hk in keepers:
        d = asdict(hk)
        d["skills"] = "|".join(hk.skills)
        d["languages"] = "|".join([x.lower() for x in hk.languages])
        d["blocked_ranges"] = "|".join([f"{s}:{e}" for s, e in hk.blocked_ranges])
        hk_rows.append(d)

    hk_df = pd.DataFrame(hk_rows)
    hk_df.to_csv("artifacts/housekeepers.csv", index=False)

    # Recommendation training
    pairs = generate_training_pairs(keepers, n_queries=1400, pairs_per_query=30, seed=42)
    pairs.to_csv("artifacts/training_pairs.csv", index=False)

    feature_cols = [
        "pref_min_age", "pref_budget", "pref_skill_count",
        "hk_age", "hk_experience_years", "hk_price",
        "skill_overlap", "age_ok", "gender_ok", "city_match", "budget_ok",
        "lang_overlap", "language_ok",
        "availability_ok", "overlap_days",
    ]

    X = pairs[feature_cols].astype(float)
    y = pairs["label_match"].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    reco = LogisticRegression(max_iter=300)
    reco.fit(Xtr, ytr)

    print("\n=== Recommender model report ===")
    print(classification_report(yte, reco.predict(Xte), digits=3))

    dump(reco, "artifacts/reco_model.joblib")
    dump(feature_cols, "artifacts/reco_features.joblib")

    # Intent training (multiclass)
    intent_df = generate_intent_dataset(seed=42)
    X_text = intent_df["text"].values
    y_int = intent_df["intent"].values

    Xtr_t, Xte_t, ytr_t, yte_t = train_test_split(X_text, y_int, test_size=0.2, random_state=42, stratify=y_int)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr_vec = vec.fit_transform(Xtr_t)
    Xte_vec = vec.transform(Xte_t)

    intent_model = LogisticRegression(max_iter=400)
    intent_model.fit(Xtr_vec, ytr_t)

    print("\n=== Intent model report ===")
    print(classification_report(yte_t, intent_model.predict(Xte_vec), digits=3))

    dump(vec, "artifacts/intent_vectorizer.joblib")
    dump(intent_model, "artifacts/intent_model.joblib")

    print("\nSaved artifacts to ./artifacts")
    print("Next: uvicorn app:app --reload")


if __name__ == "__main__":
    main()
