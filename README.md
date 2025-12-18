# Casaligan AI (`casaligan_ai`)

Casaligan AI is a demo **AI component** for the Casaligan platform (houseowner ↔ housekeeper matching).
It provides a **chat-based assistant** that recommends housekeepers using natural-language requests like:

- `female chavacano cleaning near Tetuan available on 2026-01-12 budget 650`
- `baby care from Jan 12 to Jan 14 near Zamboanga City`
- `bisaya deep cleaning under 200 near Guiwan`

---

## What this project does

Casaligan Assistant:
1) Detects user intent (recommendations vs support/help)
2) Extracts preferences from chat text (skills, language, location, budget, dates, etc.)
3) Filters housekeepers by availability using blocked calendar date ranges
4) Ranks and returns recommended profiles
5) Shows a UI with suggestion cards + profile modal + packages

---

## Features

- Chat-based housekeeper search (no long forms)
- Intent handling (recommendations vs support/job-post help)
- Preference extraction (skills, language, location, budget, dates)
- Availability filtering (blocked date ranges)
- Ranked suggestions with compatibility score
- Profile modal with service packages

---

## Tech Stack

- **FastAPI** (backend API + server)
- **scikit-learn** (Logistic Regression models)
- **HTML + Tailwind CDN + JavaScript** (frontend demo UI)

---

## Project Structure

> Tip: `artifacts/` is required to run (contains trained models + dataset).

```txt
app.py
requirements.txt
README.md
.gitignore
artifacts/   (models + dataset)
static/      (UI files)
```

---

## Setup and Installation

### 1) Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/casaligan_ai.git
cd casaligan_ai
```

### 2) Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

#### Windows (CMD)

```cmd
python -m venv venv
venv\Scripts\activate
```

#### WSL / Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run the server

```bash
uvicorn app:app --reload
```

Open in browser:

- http://127.0.0.1:8000

---

## How to Use (Demo)

1) Type a request in the chat (skills + language + location + date + budget).
2) The assistant replies with recommended housekeepers.
3) Click a card to view profile details and packages.

Example queries:
- `female chavacano cleaning near Tetuan available on 2026-01-12 budget 650`
- `baby care from Jan 12 to Jan 14 near Zamboanga City`
- `bisaya deep cleaning under 200 near Guiwan`
- `i can't login` (should return support help, not recommendations)

---

## Required Files

These must exist for the demo to work:

- `artifacts/housekeepers.csv`
- `artifacts/reco_model.joblib`
- `artifacts/reco_features.joblib`
- `artifacts/intent_vectorizer.joblib`
- `artifacts/intent_model.joblib`

---

## Troubleshooting

### UI changes not showing
Hard refresh: **Ctrl + Shift + R**

### `uvicorn` not found
Activate the virtual environment and install requirements again:

```bash
pip install -r requirements.txt
```

### Missing artifacts
Make sure `artifacts/` is committed and not ignored.

---

## License

For school/demo use.
