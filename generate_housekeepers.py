import os
import random
from datetime import date, timedelta

import pandas as pd

random.seed(42)

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

# Reference-grounded-ish skill categories (use TESDA/ILO/O*NET in documentation)
SKILLS = [
    "cleaning", "deep cleaning", "laundry", "ironing", "cooking",
    "baby care", "elder care", "pet care", "gardening", "house organization"
]
LANGUAGES = ["english", "filipino", "chavacano", "bisaya"]
CITIES = ["Zamboanga City", "Tetuan", "Guiwan", "Putik", "Canelar", "Divisoria", "Ayala", "Sta. Maria", "Sangali"]
GENDERS = ["female", "male"]
PACKAGE_TYPES = ["hourly", "daily", "weekly"]

NAMES = [
    "Aira Santos", "Janelle Cruz", "Maricel Garcia", "Rhea Fernandez", "Diana Lopez",
    "Kyla Ramos", "Nicole Reyes", "Hazel Torres", "Mikaela Flores", "Jasmine Aquino",
    "John Dela Cruz", "Mark Villanueva", "Paolo Mendoza", "Carlo Navarro", "Ryan Bautista",
    "Liza Perez", "Mira Castillo", "Sheila Domingo", "Ana Marie Lim", "Cherry Salazar",
]

def random_blocked_ranges():
    # 0 to 3 blocked ranges like "2026-01-10:2026-01-12|2026-02-02:2026-02-03"
    k = random.randint(0, 3)
    ranges = []
    base = date.today()
    for _ in range(k):
        start = base + timedelta(days=random.randint(1, 60))
        end = start + timedelta(days=random.randint(0, 5))
        ranges.append(f"{start.isoformat()}:{end.isoformat()}")
    return "|".join(ranges)

rows = []
for i in range(1, 81):  # 80 housekeepers (change if you want)
    name = random.choice(NAMES)
    age = random.randint(20, 45)
    gender = random.choice(GENDERS)
    city = random.choice(CITIES)

    skills = random.sample(SKILLS, k=random.randint(2, 4))
    languages = random.sample(LANGUAGES, k=random.randint(1, 3))

    experience_years = random.randint(1, 12)
    package_type = random.choice(PACKAGE_TYPES)

    # Base price depends on package type (demo)
    if package_type == "hourly":
        price = random.randint(80, 180)
    elif package_type == "daily":
        price = random.randint(350, 900)
    else:
        price = random.randint(1800, 4500)

    blocked_ranges = random_blocked_ranges()

    rows.append({
        "housekeeper_id": f"HK-{i:04d}",
        "name": name,
        "age": age,
        "gender": gender,
        "city": city,
        "skills": "|".join(skills),
        "languages": "|".join(languages),
        "price": price,
        "experience_years": experience_years,
        "blocked_ranges": blocked_ranges,
        "package_type": package_type,
        # optional fields (display-only if you use them)
        "religion": random.choice(["catholic", "islam", "christian", "none"]),
    })

df = pd.DataFrame(rows)
out_path = os.path.join(ART, "housekeepers.csv")
df.to_csv(out_path, index=False, encoding="utf-8")

print("✅ Created:", out_path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
