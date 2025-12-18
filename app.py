import os
import re
import random
from typing import Dict, Any, List, Tuple, Optional
from datetime import date, timedelta

import pandas as pd
from joblib import load
from dateutil.parser import parse as dt_parse

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


ART = "artifacts"
random.seed(42)

HK = pd.read_csv(os.path.join(ART, "housekeepers.csv"))

RECO = load(os.path.join(ART, "reco_model.joblib"))
RECO_FEATURES: List[str] = load(os.path.join(ART, "reco_features.joblib"))

VEC = load(os.path.join(ART, "intent_vectorizer.joblib"))
INTENT_MODEL = load(os.path.join(ART, "intent_model.joblib"))

# Reference-grounded taxonomy (skills/categories). (Cite TESDA/ILO/O*NET in documentation)
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

DOMAIN_KEYWORDS = [
    "housekeeper", "kasambahay", "helper", "maid", "yaya",
    "recommend", "suggest", "find", "hire", "match",
    "cleaning", "laundry", "ironing", "cooking", "baby", "elder",
    "available", "schedule", "budget", "near", "location", "package"
]

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

# ---------- Intent detection ----------
def detect_intent(text: str) -> str:
    t = text.lower().strip()

    # Strong rule override for recommendation intent
    if any(k in t for k in ["recommend", "suggest", "find", "hire", "housekeeper", "kasambahay", "maid", "yaya"]):
        return "RECOMMEND"

    # If user message looks clearly out-of-domain, classify as OFF_TOPIC quickly
    if not any(k in t for k in DOMAIN_KEYWORDS) and len(t.split()) >= 4:
        # Let ML model still decide, but bias towards OFF_TOPIC if it predicts weirdly
        X = VEC.transform([text])
        pred = INTENT_MODEL.predict(X)[0]
        return pred if pred in ["SUPPORT", "SMALLTALK", "JOB_POST"] else "OFF_TOPIC"

    X = VEC.transform([text])
    pred = INTENT_MODEL.predict(X)[0]
    return pred

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

# ---------- Reco features ----------
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

def _pick(lines: List[str]) -> str:
    return random.choice(lines)

def _format_summary(pref: Dict[str, Any]) -> str:
    bits = []
    if pref.get("skills"):
        bits.append("skills: " + ", ".join(pref["skills"]))
    if pref.get("city"):
        bits.append("near " + pref["city"])
    if pref.get("language"):
        bits.append("language: " + pref["language"])
    if pref.get("date_start"):
        if pref.get("date_end") and pref["date_end"] != pref["date_start"]:
            bits.append(f"dates: {pref['date_start']}–{pref['date_end']}")
        else:
            bits.append("date: " + pref["date_start"])
    if pref.get("budget") is not None:
        bits.append(f"budget ≤ {pref['budget']}")
    return ", ".join(bits)

def ask_missing(pref: Dict[str, Any]) -> Optional[str]:
    # Instead of a rigid template, give a short natural nudge
    has_work_info = bool(pref["skills"]) or bool(pref["date_start"]) or bool(pref["city"]) or (pref["budget"] is not None)
    if not has_work_info:
        return _pick([
            "Got you 😊 What do you need help with most—cleaning, cooking, baby care, laundry, etc.?",
            "Sure! What’s the main task you need? (cleaning / cooking / baby care / laundry)",
            "Okay—tell me the top task you need, and if you have a date or location, add it too."
        ]) + "\n\nTip: you can type like `cleaning near Tetuan available on 2026-01-12 budget 650`."
    return None

def recommend(pref: Dict[str, Any], top_k=10) -> List[Dict[str, Any]]:
    rows = []
    for _, hk in HK.iterrows():
        feats = build_features(pref, hk)

        # Hard filters to keep results relevant
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

        Xdf = pd.DataFrame([[feats[c] for c in RECO_FEATURES]], columns=RECO_FEATURES)
        model_proba = float(RECO.predict_proba(Xdf)[0][1])

        # Compatibility score based on satisfied constraints
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
        pkg = hk.get("package_type", "hourly")
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
            "package_type": pkg,
            "base_price": price,
            "match_score": compat_score,
            "model_score": round(model_proba * 100, 1),
            "packages": packages
        })

    rows.sort(key=lambda x: x["model_score"], reverse=True)
    return rows[:top_k]

# ---------- Natural replies ----------
def intro() -> str:
    return _pick([
        "Hi! I’m Casaligan Assistant 🤝 Tell me what you need and I’ll find matches.",
        "Hello 😊 Describe the housekeeper you need and I’ll recommend options.",
        "Hey! Tell me your needs (skills, date, budget, location) and I’ll match you."
    ]) + "\n\nTip: `cleaning near Tetuan available on 2026-01-12 budget 650`"

def reply_off_topic() -> str:
    return _pick([
        "I can help with Casaligan matching 😊 If you want housekeeper suggestions, tell me the task + location/date/budget.",
        "I might not be the best for that 😅 But I *can* recommend housekeepers—just say what you need (skills + date + location).",
        "Let’s keep it Casaligan-related 🙌 Tell me: what task do you need help with (cleaning, cooking, baby care, etc.)?"
    ]) + "\n\nQuick example: `baby care from Jan 12 to Jan 14 near Zamboanga City`"

def reply_support(msg: str) -> str:
    m = msg.lower()
    if any(x in m for x in ["login", "password", "reset"]):
        return (
            "Okay—let’s fix that.\n"
            "• Double-check your email/phone spelling\n"
            "• Try “Forgot Password”\n"
            "• If it still fails, tell me the exact error text you see\n\n"
            "Tip: after login works, you can chat: `cleaning near Tetuan budget 650`."
        )
    if any(x in m for x in ["payment", "refund"]):
        return (
            "Got it. For payments:\n"
            "• Check internet and balance\n"
            "• Try again or change payment method\n"
            "• Save the reference number if shown\n\n"
            "Tell me what step failed (pay button, verification, or confirmation)."
        )
    return (
        "Tell me what’s happening (login, payment, profile view, or app error) and what the message says.\n"
        "I’ll guide you step-by-step."
    )

def reply_job_post() -> str:
    return (
        "Sure—here’s a quick way to post a job:\n"
        "1) Location\n"
        "2) Tasks needed (skills)\n"
        "3) Schedule (date/range)\n"
        "4) Budget\n\n"
        "Example: `Tetuan, cleaning + laundry, Jan 12–14, budget 650`"
    )

def reply_reco_results(pref: Dict[str, Any], n: int) -> str:
    summ = _format_summary(pref)
    lead = _pick([
        "Alright—here are the best matches I found 😊",
        "Okay! Here are the closest matches 👇",
        "Got it. These are the best options based on what you said:"
    ])
    if summ:
        lead += f"\n({summ})"
    lead += f"\n\nTip: Click a profile card to view packages. You can also add more filters like `chavacano`, `under 500`, or a date."
    return lead

def reply_no_results(pref: Dict[str, Any]) -> str:
    summ = _format_summary(pref)
    msg = _pick([
        "Hmm… I couldn’t find a strong match with those exact constraints.",
        "No exact match popped up for that combo.",
        "I tried—but your filters are a bit tight for the current list."
    ])
    if summ:
        msg += f"\n({summ})"
    msg += "\n\nTry one of these:\n• Remove 1 filter (budget/date/language)\n• Use a nearby location\n• Add only 1 main skill first (ex: `cleaning`) then refine"
    return msg

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

    if not msg:
        return JSONResponse({
            "reply": "Type a message and I’ll help 😊",
            "intent": "SMALLTALK",
            "needs_more_info": False,
            "preferences": {},
            "results": []
        })

    # Follow-up mode: if we were already gathering info, treat message as recommendation refinement
    followup_mode = bool(ctx.get("needs_more_info")) or (ctx.get("last_intent") == "RECOMMEND")

    intent = detect_intent(msg)
    if followup_mode and intent in ["SMALLTALK", "OFF_TOPIC"]:
        intent = "RECOMMEND"

    if intent == "RECOMMEND":
        pref_new = extract_preferences(msg)
        if isinstance(ctx.get("preferences"), dict) and ctx.get("preferences"):
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
                "results": [],
                "last_intent": "RECOMMEND"
            })

        results = recommend(pref, top_k=10)
        if not results:
            return JSONResponse({
                "reply": reply_no_results(pref),
                "intent": intent,
                "needs_more_info": False,
                "preferences": pref,
                "results": [],
                "last_intent": "RECOMMEND"
            })

        return JSONResponse({
            "reply": reply_reco_results(pref, len(results)),
            "intent": intent,
            "needs_more_info": False,
            "preferences": pref,
            "results": results,
            "last_intent": "RECOMMEND"
        })

    # Non-recommendation intents
    if intent == "SMALLTALK":
        return JSONResponse({
            "reply": intro(),
            "intent": intent,
            "needs_more_info": False,
            "preferences": {},
            "results": [],
            "last_intent": "SMALLTALK"
        })

    if intent == "SUPPORT":
        return JSONResponse({
            "reply": reply_support(msg),
            "intent": intent,
            "needs_more_info": False,
            "preferences": {},
            "results": [],
            "last_intent": "SUPPORT"
        })

    if intent == "JOB_POST":
        return JSONResponse({
            "reply": reply_job_post(),
            "intent": intent,
            "needs_more_info": False,
            "preferences": {},
            "results": [],
            "last_intent": "JOB_POST"
        })

    # OFF_TOPIC
    return JSONResponse({
        "reply": reply_off_topic(),
        "intent": "OFF_TOPIC",
        "needs_more_info": False,
        "preferences": {},
        "results": [],
        "last_intent": "OFF_TOPIC"
    })
