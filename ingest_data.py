import ollama
import json
import os
import re
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_FILE = 'data/raw/my_messy_workouts.txt'
PROCESSED_FILE = 'data/processed/exercise_db.json'
INGESTION_MODEL = 'qwen2.5'  # Fast, lightweight model for structured data extraction

# Matches the date header line: "4/9/2026: shoulders/traps"
DATE_HEADER_RE = re.compile(r'^(\d{1,2}/\d{1,2}/\d{4})\s*:\s*(.+)$')

# Lines that should always be skipped — not exercises
SKIP_PATTERNS = [
    re.compile(r'^\d+\s*min\s*(stretch|warmup|cool)', re.IGNORECASE),  # "15 min stretch"
    re.compile(r'^stretch', re.IGNORECASE),
]


def parse_date(date_str):
    """Parse M/D/YYYY into ISO-8601. Returns None on failure."""
    try:
        return datetime.strptime(date_str.strip(), "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def clean_n(v):
    """Extract the first integer from a value. Returns 0 if none found."""
    n = re.findall(r'\d+', str(v))
    return int(n[0]) if n else 0


def should_skip(line):
    """Return True if this line should never be sent to the model."""
    return any(p.match(line) for p in SKIP_PATTERNS)


def split_into_days(content):
    """
    Split the raw text into day blocks using date header lines as boundaries.
    Returns a list of dicts: {date, session, lines}
    """
    days = []
    current_day = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        date_match = DATE_HEADER_RE.match(line)
        if date_match:
            # Save the previous day before starting a new one
            if current_day:
                days.append(current_day)
            current_day = {
                "date": parse_date(date_match.group(1)),
                "session": date_match.group(2).strip(),
                "lines": []
            }
        elif current_day is not None:
            current_day["lines"].append(line)

    # Don't forget the last day
    if current_day:
        days.append(current_day)

    return days


def clean_workout_data():
    if not os.path.exists(RAW_FILE):
        print(f"ERROR: Raw file not found at '{RAW_FILE}'. Aborting.")
        return

    with open(RAW_FILE, 'r') as f:
        content = f.read()

    days = split_into_days(content)
    print(f"Found {len(days)} workout days.\n")
    print(f"Using {INGESTION_MODEL} with Brute Force logic...")

    master_database = []
    failed_lines = []

    for i, day in enumerate(days):
        day_id = i + 1
        workout_date = day["date"]
        session_label = day["session"]

        print(f"\n--- Day {day_id} | {workout_date} | {session_label} ---")

        for line in day["lines"]:
            # Skip stretch/warmup-only lines
            if should_skip(line):
                print(f"  ⏭  Skipped: '{line}'")
                continue

            prompt = (
                f"Convert this workout line into JSON with exactly these keys: "
                f"exercise_name (string), weight_lbs (number, 0 if bodyweight or not applicable), "
                f"sets (number, 0 if not applicable), reps (number, 0 if not applicable), "
                f"notes (string, capture duration/speed/incline info here, empty string if none). "
                f"Return only valid JSON, no explanation. Line: \"{line}\""
            )

            try:
                response = ollama.chat(
                    model=INGESTION_MODEL,
                    format='json',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0}
                )

                ex = json.loads(response['message']['content'])

                master_database.append({
                    "date": workout_date,
                    "day_id": day_id,
                    "session": session_label,
                    "exercise_name": ex.get('exercise_name', 'Unknown'),
                    "weight_lbs": clean_n(ex.get('weight_lbs')),
                    "sets": clean_n(ex.get('sets')),
                    "reps": clean_n(ex.get('reps')),
                    "notes": ex.get('notes', ''),
                })
                print(f"  ✅ {ex.get('exercise_name')}")

            except json.JSONDecodeError as e:
                msg = f"  ⚠️  JSON parse failed | Day {day_id} | Line: '{line}' | Error: {e}"
                print(msg)
                failed_lines.append({"day_id": day_id, "date": workout_date, "line": line, "error": str(e)})
            except Exception as e:
                msg = f"  ❌ Unexpected error | Day {day_id} | Line: '{line}' | Error: {e}"
                print(msg)
                failed_lines.append({"day_id": day_id, "date": workout_date, "line": line, "error": str(e)})

    # Save main database
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(master_database, f, indent=4)
    print(f"\n✅ SUCCESS! Total exercises captured: {len(master_database)}")

    # Save failed lines log
    if failed_lines:
        failed_path = PROCESSED_FILE.replace('.json', '_failed.json')
        with open(failed_path, 'w') as f:
            json.dump(failed_lines, f, indent=4)
        print(f"⚠️  {len(failed_lines)} lines failed to parse. Review: {failed_path}")
    else:
        print("✅ Zero parse failures.")


if __name__ == "__main__":
    clean_workout_data()