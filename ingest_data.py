import ollama
import json
import os
import re

# CONFIG
RAW_FILE = 'data/raw/my_messy_workouts.txt'
PROCESSED_FILE = 'data/processed/exercise_db.json'
MODEL_NAME = 'llama3.2'


def clean_workout_data():
    if not os.path.exists(RAW_FILE): return

    with open(RAW_FILE, 'r') as f:
        content = f.read()

    days = [d.strip() for d in content.split('---') if d.strip() and not d.strip().startswith('DAY')]
    master_database = []

    # WE USE LLAMA 3.2 FOR SPEED SINCE THE TASK IS NOW SIMPLE
    model = MODEL_NAME
    print(f"Using {model} with Brute Force logic...")

    for i, day_text in enumerate(days):
        print(f"--- Day {i + 1} ---")

        # Split the day into individual lines
        lines = [l.strip() for l in day_text.split('\n') if len(l.strip()) > 5]

        for line in lines:
            # Skip the "Solid energy" style flavor text
            if ":" not in line and "x" not in line:
                continue

            prompt = f"Convert this workout line into JSON: '{{ \"exercise_name\": \"...\", \"weight_lbs\": 0, \"sets\": 0, \"reps\": 0, \"notes\": \"...\" }}'. Line: {line}"

            try:
                response = ollama.chat(
                    model=model,
                    format='json',
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0}
                )

                ex = json.loads(response['message']['content'])

                # Cleanup the numbers
                def clean_n(v):
                    n = re.findall(r'\d+', str(v))
                    return int(n[0]) if n else 0

                master_database.append({
                    "exercise_name": ex.get('exercise_name', 'Unknown'),
                    "weight_lbs": clean_n(ex.get('weight_lbs')),
                    "sets": clean_n(ex.get('sets')),
                    "reps": clean_n(ex.get('reps')),
                    "notes": ex.get('notes', ''),
                    "day_id": i + 1
                })
                print(f"  Processed: {ex.get('exercise_name')}")
            except:
                continue

    # Final Save
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    with open(PROCESSED_FILE, 'w') as f:
        json.dump(master_database, f, indent=4)
    print(f"\nSUCCESS! Total exercises captured: {len(master_database)}")


if __name__ == "__main__":
    clean_workout_data()