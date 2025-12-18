import os
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import date, timedelta

import numpy as np
import pandas as pd
from joblib import load
from dateutil.parser import parse as dt_parse

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ART = "artifacts"

HK = pd.read_csv(os.path.join(ART, "housekeepers.csv"))

RECO = load(os.path.join(ART, "reco_model.joblib"))
RECO_FEATURES: List[str] = load(os.path.join(ART, "reco_features.joblib"))

VEC = load(os.path.join(ART, "intent_vectorizer.joblib"))
INTENT_MODEL = load(os.path.join(ART, "intent_model.joblib"))


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


# ---------- Availability helpers ----------
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


# ---------- Intent detection with rule override ----------
def detect_intent(text: str) -> str:
    t = text.lower()

    recommend_keywords = [
        "housekeeper", "kasambahay", "helper", "maid", "yaya",
        "suggest", "recommend", "find", "looking for", "hire", "match"
    ]
    if any(k in t for k in recommend_keywords):
        return "RECOMMEND"

    X = VEC.transform([text])
    return INTENT_MODEL.predict(X)[0]


# ---------- Slot extraction ----------
def _find_city(t: str) -> Optional[str]:
    for c in CITIES:
        if c.lower() in t:
            return c
    return None


def _find_skills(t: str) -> List[str]:
    found = []
    for s in SKILLS:
        if s in t:
            found.append(s)
    if "babycare" in t:
        found.append("baby care")
    return list(dict.fromkeys(found))


def _find_language(t: str) -> Optional[str]:
    if "chabacano" in t or "zamboangueño" in t:
        return "chavacano"
    for lang in LANGUAGES:
        if re.search(rf"\b{lang}\b", t):
            return lang
    return None


def _find_gender(t: str) -> Optional[str]:
    for g in GENDERS:
        if re.search(rf"\b{g}\b", t):
            return g
    return None


def _find_min_age(t: str) -> Optional[int]:
    m = re.search(r"\b(above|over|at least|min)\s+(\d{2})\b", t)
    if m:
        return int(m.group(2))
    m2 = re.search(r"\b(\d{2})\s*\+\b", t)
    if m2:
        return int(m2.group(1))
    return None


def _find_budget(t: str) -> Optional[int]:
    m = re.search(r"\b(budget|max|under|below)\s+(\d{2,5})\b", t)
    if m:
        return int(m.group(2))
    return None


def _parse_date_any(s: str) -> Optional[date]:
    try:
        d = dt_parse(s, fuzzy=True, default=dt_parse(date.today().isoformat())).date()
        return d
    except Exception:
        return None


def _find_date_range(t: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"\bfrom\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        return m.group(1), m.group(2)

    m = re.search(r"\b(on|available on|available)\s+(\d{4}-\d{2}-\d{2})\b", t)
    if m:
        d = m.group(2)
        return d, d

    if "tomorrow" in t:
        d = (date.today() + timedelta(days=1)).isoformat()
        return d, d
    if "today" in t:
        d = date.today().isoformat()
        return d, d

    m = re.search(r"\bfrom\s+([a-z]{3,9}\s+\d{1,2})\s+to\s+([a-z]{3,9}\s+\d{1,2})\b", t)
    if m:
        d1 = _parse_date_any(m.group(1))
        d2 = _parse_date_any(m.group(2))
        if d1 and d2:
            return d1.isoformat(), d2.isoformat()

    m = re.search(r"\b(on|available on|available)\s+([a-z]{3,9}\s+\d{1,2})\b", t)
    if m:
        d1 = _parse_date_any(m.group(2))
        if d1:
            ds = d1.isoformat()
            return ds, ds

    return None, None


def extract_preferences(text: str) -> Dict[str, Any]:
    t = text.lower().strip()
    ds, de = _find_date_range(t)

    return {
        "gender": _find_gender(t),
        "min_age": _find_min_age(t),
        "language": _find_language(t),
        "city": _find_city(t),
        "skills": _find_skills(t),
        "budget": _find_budget(t),
        "date_start": ds,
        "date_end": de,
    }


def merge_preferences(base: Dict[str, Any], newp: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)

    for k in ["gender", "min_age", "language", "city", "budget", "date_start", "date_end"]:
        if newp.get(k) is not None:
            merged[k] = newp[k]

    base_skills = merged.get("skills") or []
    new_skills = newp.get("skills") or []
    merged["skills"] = list(dict.fromkeys(base_skills + new_skills))

    return merged


# ---------- Recommender feature building ----------
def build_features(pref: Dict[str, Any], hk_row: pd.Series) -> Dict[str, float]:
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


def ask_missing(pref: Dict[str, Any]) -> Optional[str]:
    has_work_info = bool(pref["skills"]) or bool(pref["date_start"]) or bool(pref["city"]) or (pref["budget"] is not None)
    if not has_work_info:
        return (
            "Sure! I can do that 😊\n"
            "Quick question so I can match better:\n"
            "• What tasks do you need most? (cleaning, cooking, baby care, etc.)\n"
            "Optional: location, date, and budget."
        )
    return None


def recommend(pref: Dict[str, Any], top_k=10) -> List[Dict[str, Any]]:
    rows = []

    for _, hk in HK.iterrows():
        feats = build_features(pref, hk)

        # Hard filters
        if pref["skills"] and feats["skill_overlap"] < 1:
            continue
        if pref["language"] and feats["language_ok"] == 0:
            continue
        if pref["date_start"] and feats["availability_ok"] == 0:
            continue
        if pref["budget"] is not None and feats["budget_ok"] == 0:
            continue
        if pref["gender"] is not None and feats["gender_ok"] == 0:
            continue
        if pref["min_age"] is not None and feats["age_ok"] == 0:
            continue

        # ✅ FIX sklearn warning: predict with DataFrame containing feature names
        Xdf = pd.DataFrame([[feats[c] for c in RECO_FEATURES]], columns=RECO_FEATURES)
        model_proba = float(RECO.predict_proba(Xdf)[0][1])

        # Compatibility score (display)
        constraint_count = 0
        satisfied = 0

        if pref["gender"] is not None:
            constraint_count += 1; satisfied += int(feats["gender_ok"] == 1)
        if pref["language"] is not None:
            constraint_count += 1; satisfied += int(feats["language_ok"] == 1)
        if pref["min_age"] is not None:
            constraint_count += 1; satisfied += int(feats["age_ok"] == 1)
        if pref["city"] is not None:
            constraint_count += 1; satisfied += int(feats["city_match"] == 1)
        if pref["budget"] is not None:
            constraint_count += 1; satisfied += int(feats["budget_ok"] == 1)
        if pref["skills"]:
            constraint_count += 1; satisfied += int(feats["skill_overlap"] >= 1)
        if pref["date_start"]:
            constraint_count += 1; satisfied += int(feats["availability_ok"] == 1)

        compat_score = round((satisfied / max(constraint_count, 1)) * 100, 1)

        # packages demo
        pkg = hk["package_type"]
        price = int(hk["price"])
        if pkg == "hourly":
            packages = [
                {"name": "Basic (2 hrs)", "price": max(150, price * 2)},
                {"name": "Standard (4 hrs)", "price": max(300, price * 4)},
                {"name": "Deep Clean (6 hrs)", "price": max(450, price * 6)},
            ]
        elif pkg == "daily":
            packages = [
                {"name": "Day Shift", "price": price},
                {"name": "Day + Cooking", "price": int(price * 1.15)},
                {"name": "Full Service", "price": int(price * 1.25)},
            ]
        else:
            packages = [
                {"name": "Week Basic", "price": price},
                {"name": "Week Standard", "price": int(price * 1.12)},
                {"name": "Week Premium", "price": int(price * 1.25)},
            ]

        rows.append({
            "housekeeper_id": hk["housekeeper_id"],
            "name": hk["name"],
            "age": int(hk["age"]),
            "gender": hk["gender"],
            "city": hk["city"],
            "skills": str(hk["skills"]).split("|"),
            "languages": str(hk["languages"]).split("|"),
            "experience_years": int(hk["experience_years"]),
            "package_type": hk["package_type"],
            "base_price": price,
            "match_score": compat_score,
            "model_score": round(model_proba * 100, 1),
            "packages": packages
        })

    rows.sort(key=lambda x: x["model_score"], reverse=True)
    return rows[:top_k]


# ---------- Conversational routing ----------
def friendly_intro():
    return (
        "Hi! I’m Casaligan Assistant 🤝\n"
        "Tell me what you need naturally, like:\n"
        "• `female chavacano cleaning near Tetuan available on 2026-01-12 budget 650`\n"
        "• `baby care from Jan 12 to Jan 14 near Zamboanga City`\n\n"
        "You can mention: skills, budget, language, location, age, gender, and dates."
    )


def handle_non_reco(intent: str, msg: str) -> str:
    m = msg.lower()

    if intent == "SMALLTALK":
        if any(x in m for x in ["hello", "hi", "good morning", "good evening"]):
            return friendly_intro()
        if "who are you" in m or "what can you do" in m:
            return "I’m Casaligan Assistant 🙂 I help you find available housekeepers fast through chat."
        if "thanks" in m or "thank you" in m:
            return "You’re welcome! Want me to recommend someone now? 😊"
        return friendly_intro()

    if intent == "JOB_POST":
        return (
            "Sure! For a strong job post, give me these:\n"
            "1) Location (city/area)\n"
            "2) Tasks (skills)\n"
            "3) Schedule (date/range)\n"
            "4) Budget\n\n"
            "Example: `Tetuan, cleaning + laundry, Jan 12–14, budget 650`"
        )

    if intent == "SUPPORT":
        if "login" in m or "password" in m or "reset" in m:
            return (
                "For login issues:\n"
                "• Double-check email/phone spelling\n"
                "• Use “Forgot Password”\n"
                "• If locked, wait a bit then retry\n\n"
                "Tell me the exact error and I’ll guide you."
            )
        if "payment" in m or "refund" in m:
            return (
                "For payments/refunds:\n"
                "• Check balance and internet\n"
                "• Retry or switch method\n"
                "• Save the reference number\n\n"
                "Tell me what step failed and I’ll help."
            )
        return "Tell me what went wrong (login/payment/app navigation) and I’ll help."

    return (
        "I can help with:\n"
        "1) Finding housekeepers\n"
        "2) Posting a job\n"
        "3) App support\n\n"
        "If you want recommendations, describe skills + date + location."
    )


# ---------- FastAPI ----------
app = FastAPI(title="Casaligan AI Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatIn(BaseModel):
    message: str
    context: Dict[str, Any] = {}


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")


@app.post("/api/chat")
def chat(payload: ChatIn):
    msg = (payload.message or "").strip()
    ctx = payload.context or {}
    followup_mode = bool(ctx.get("needs_more_info"))

    if not msg:
        return JSONResponse({
            "reply": "Type a message and I’ll help 😊",
            "intent": "SMALLTALK",
            "needs_more_info": False,
            "preferences": {},
            "results": []
        })

    intent = detect_intent(msg)

    if followup_mode:
        intent = "RECOMMEND"

    if intent == "RECOMMEND":
        pref_new = extract_preferences(msg)

        if followup_mode and isinstance(ctx.get("preferences"), dict):
            pref = merge_preferences(ctx["preferences"], pref_new)
        else:
            pref = pref_new

        missing = ask_missing(pref)
        if missing:
            return JSONResponse({
                "reply": missing,
                "intent": intent,
                "needs_more_info": True,
                "preferences": pref,
                "results": []
            })

        results = recommend(pref, top_k=10)

        if not results:
            reply = (
                "Hmm… I couldn’t find a strong match with those exact constraints.\n"
                "Try relaxing 1 filter (budget/date/language) or add a nearby location."
            )
            return JSONResponse({
                "reply": reply,
                "intent": intent,
                "needs_more_info": False,
                "preferences": pref,
                "results": []
            })

        summary_bits = []
        if pref["skills"]:
            summary_bits.append(f"skills: {', '.join(pref['skills'])}")
        if pref["city"]:
            summary_bits.append(f"near {pref['city']}")
        if pref["language"]:
            summary_bits.append(f"language: {pref['language']}")
        if pref["date_start"]:
            summary_bits.append(
                f"date: {pref['date_start']}" if pref["date_start"] == pref["date_end"]
                else f"dates: {pref['date_start']}–{pref['date_end']}"
            )
        if pref["budget"] is not None:
            summary_bits.append(f"budget ≤ {pref['budget']}")

        reply = "Alright — here are the best matches I found"
        if summary_bits:
            reply += " (" + ", ".join(summary_bits) + ")"
        reply += ". Click a profile to view details and packages."

        return JSONResponse({
            "reply": reply,
            "intent": intent,
            "needs_more_info": False,
            "preferences": pref,
            "results": results
        })

    reply = handle_non_reco(intent, msg)
    return JSONResponse({
        "reply": reply,
        "intent": intent,
        "needs_more_info": False,
        "preferences": {},
        "results": []
    })
