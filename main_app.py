from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import date, datetime

# ── MODEL CONFIG ──────────────────────────────────────────────────────────────
# Backend is controlled via secrets.toml — no code changes needed when switching
# between local dev and Streamlit Cloud deployment.
#
# secrets.toml:
#   COACH_BACKEND = "local"   → Ollama + qwen3.6  (requires local Ollama instance)
#   COACH_BACKEND = "cloud"   → Gemini 2.5 Flash Lite via Google AI API
#   GEMINI_API_KEY = "..."    → required only when COACH_BACKEND = "cloud"

COACH_NAME        = "Coach GT"
LOCAL_COACH_MODEL = 'qwen3.6'
CLOUD_COACH_MODEL = 'gemini-2.5-flash-lite'

# ── FILE PATHS ────────────────────────────────────────────────────────────────
DB_PATH = 'data/processed/exercise_db.json'
RAW_FILE = 'data/raw/my_messy_workouts.txt'

# ── PROGRESSIVE OVERLOAD CONFIG ───────────────────────────────────────────────
STAGNATION_THRESHOLD = 3  # Sessions without a new PR before flagging stagnation
HISTORY_DAYS = 20  # Max days to show in the workout history view

# ── INGESTION CONFIG (mirrors ingest_data.py) ─────────────────────────────────
INGESTION_MODEL = 'qwen2.5'
DATE_HEADER_RE = re.compile(r'^(\d{1,2}/\d{1,2}/\d{4})\s*:\s*(.+)$')
SKIP_PATTERNS = [
    re.compile(r'^\d+\s*min\s*(stretch|warmup|cool)', re.IGNORECASE),
    re.compile(r'^stretch', re.IGNORECASE),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Elite Fitness Tracker & Coach", layout="wide")


# ═══════════════════════════════════════════════════════════════════════════════
# PASSWORD GATE
# ═══════════════════════════════════════════════════════════════════════════════
# Password is stored in .streamlit/secrets.toml (local) and Streamlit Cloud
# secrets manager (production). Never hardcode it here.
#
# .streamlit/secrets.toml format:
#   APP_PASSWORD = "your_password_here"

def check_password():
    if st.session_state.get("authenticated"):
        return True

    st.title("🏋️‍♂️ Elite Fitness Tracker")
    st.write("Enter your password to continue.")

    pwd = st.text_input("Password", type="password", key="password_input")
    if st.button("Unlock", type="primary"):
        try:
            correct = st.secrets["APP_PASSWORD"]
        except KeyError:
            st.error("APP_PASSWORD not set. Add it to `.streamlit/secrets.toml`.")
            st.stop()

        if pwd == correct:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")

    st.stop()


if not check_password():
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_science_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
    return db


def analyze_progressive_overload(exercise_df):
    """Detect weight stagnation for a given exercise's history DataFrame."""
    sorted_df = exercise_df.sort_values('day_id', ascending=True)
    weights = [w for w in sorted_df['weight_lbs'].tolist() if w > 0]

    if len(weights) < 2:
        return {"trend": "insufficient_data", "sessions_since_pr": 0,
                "all_time_max": weights[0] if weights else 0,
                "last_weight": weights[-1] if weights else 0,
                "message": "Not enough weighted sessions to evaluate overload."}

    all_time_max = max(weights)
    last_weight = weights[-1]
    sessions_since_pr = 0
    for w in reversed(weights):
        if w < all_time_max:
            sessions_since_pr += 1
        else:
            break

    if sessions_since_pr >= STAGNATION_THRESHOLD:
        return {"trend": "stagnant", "sessions_since_pr": sessions_since_pr,
                "all_time_max": all_time_max, "last_weight": last_weight,
                "message": (f"⚠️ No weight increase in the last {sessions_since_pr} sessions "
                            f"(stuck at {last_weight} lbs, PR is {all_time_max} lbs).")}
    return {"trend": "improving", "sessions_since_pr": sessions_since_pr,
            "all_time_max": all_time_max, "last_weight": last_weight,
            "message": f"✅ Progressive overload on track. Current: {last_weight} lbs | PR: {all_time_max} lbs."}


def build_history_context(df, selected_date=None):
    """
    Always sends the coach the FULL exercise history so it can reason across
    all sessions — not just the selected day.
    """
    full_log = (
        df[['date', 'session', 'exercise_name', 'weight_lbs', 'sets', 'reps', 'notes']]
        .sort_values(['date', 'exercise_name'])
        .to_dict(orient='records')
    )
    full_log_str = f"COMPLETE WORKOUT HISTORY (all sessions):\n{json.dumps(full_log, indent=2)}"

    summary = (
        df.groupby('exercise_name')
        .agg(max_weight=('weight_lbs', 'max'), total_sets=('sets', 'sum'), sessions=('day_id', 'nunique'))
        .reset_index()
        .sort_values('total_sets', ascending=False)
    )
    summary_str = f"\nEXERCISE SUMMARY (all time, all exercises):\n{summary.to_string(index=False)}"

    if selected_date:
        day_df = df[df['date'] == selected_date]
        day_records = day_df[['exercise_name', 'weight_lbs', 'sets', 'reps', 'notes']].to_dict(orient='records')
        day_str = (
            f"\nUSER'S CURRENTLY SELECTED DAY ({selected_date}) — for reference only. "
            f"You may answer questions about any part of the full history above:\n"
            f"{json.dumps(day_records, indent=2)}"
        )
    else:
        day_str = ""

    return full_log_str + summary_str + day_str


def highlight_notes(val):
    note_text = str(val).upper()
    if any(word in note_text for word in ['STRONG', 'HEAVY', 'WEIGHTED', 'PR', 'PR!']):
        return 'background-color: #004d00; color: white'
    return ''


# ── INGESTION HELPERS (inline — no dependency on ingest_data.py) ──────────────

def clean_n(v):
    n = re.findall(r'\d+', str(v))
    return int(n[0]) if n else 0


def should_skip(line):
    return any(p.match(line) for p in SKIP_PATTERNS)


def ingest_exercise_line(line, day_id, workout_date, session_label):
    """
    Send a single exercise line to qwen2.5 and return a structured record.
    Returns None if the line should be skipped. Raises on parse errors.
    """
    if should_skip(line):
        return None

    prompt = (
        f"Convert this workout line into JSON with exactly these keys: "
        f"exercise_name (string), weight_lbs (number, 0 if bodyweight or not applicable), "
        f"sets (number, 0 if not applicable), reps (number, 0 if not applicable), "
        f"notes (string, capture duration/speed/incline info here, empty string if none). "
        f"Return only valid JSON, no explanation. Line: \"{line}\""
    )
    response = ollama.chat(
        model=INGESTION_MODEL,
        format='json',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0}
    )
    ex = json.loads(response['message']['content'])
    return {
        "date": workout_date,
        "day_id": day_id,
        "session": session_label,
        "exercise_name": ex.get('exercise_name', 'Unknown'),
        "weight_lbs": clean_n(ex.get('weight_lbs')),
        "sets": clean_n(ex.get('sets')),
        "reps": clean_n(ex.get('reps')),
        "notes": ex.get('notes', ''),
    }


def append_day_to_txt(iso_date, session_label, exercise_lines):
    """Append a new workout block to my_messy_workouts.txt in the correct format."""
    dt = datetime.strptime(iso_date, "%Y-%m-%d")
    header = f"{dt.month}/{dt.day}/{dt.year}: {session_label}"
    block = "\n" + header + "\n" + "\n".join(exercise_lines) + "\n15 min stretch\n\n"
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    with open(RAW_FILE, 'a') as f:
        f.write(block)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
st.title("🏋️‍♂️ Workout Performance Matrix")

if not os.path.exists(DB_PATH):
    st.error(f"Database not found at `{DB_PATH}`. Run `ingest_data.py` first.")
    st.stop()

with open(DB_PATH, 'r') as f:
    df = pd.DataFrame(json.load(f))

for col in ['date', 'session']:
    if col not in df.columns:
        df[col] = None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR: YOUR PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("👤 Your Profile")

user_height = st.sidebar.text_input("Height", value="5'7\"", placeholder="e.g. 6'0\"")
user_weight = st.sidebar.number_input("Body Weight (lbs)", value=155, step=1)
user_age = st.sidebar.number_input("Age", value=43, step=1)
st.sidebar.write("---")
user_goal = st.sidebar.selectbox("Primary Fitness Goal", [
    "Hypertrophy (Muscle Growth)", "Strength", "Fat Loss / Body Recomposition",
    "Muscular Endurance", "General Fitness / Health"])
user_experience = st.sidebar.selectbox("Training Experience", [
    "Beginner (< 1 year)", "Intermediate (1–3 years)", "Advanced (3+ years)"])
user_days_per_week = st.sidebar.slider("Days Available to Train Per Week", 1, 7, 4)
st.sidebar.write("---")
user_injuries = st.sidebar.text_area(
    "Injuries / Limitations",
    placeholder="e.g. Lower back strain, avoid heavy deadlifts. Right shoulder impingement.",
    height=100,
)

user_profile = {
    "height": user_height,
    "weight_lbs": user_weight,
    "age": user_age,
    "goal": user_goal,
    "experience": user_experience,
    "days_per_week": user_days_per_week,
    "injuries": user_injuries if user_injuries.strip() else "None reported",
}

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_history, tab_log, tab_coach = st.tabs(["📅 Workout History", "➕ Log Workout", f"💬 {COACH_NAME}"])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — WORKOUT HISTORY
# ───────────────────────────────────────────────────────────────────────────────
with tab_history:
    days_df = (
        df.groupby(['date', 'day_id', 'session'])
        .agg(
            exercises=('exercise_name', 'count'),
            total_sets=('sets', 'sum'),
            total_volume=('weight_lbs', lambda x: (x * df.loc[x.index, 'sets'] * df.loc[x.index, 'reps']).sum()),
        )
        .reset_index()
        .sort_values('date', ascending=False)
        .head(HISTORY_DAYS)
    )

    st.subheader(f"📅 Recent Workouts (Last {HISTORY_DAYS} Sessions)")

    if days_df.empty:
        st.warning("No workout data found. Run `ingest_data.py` first.")
    else:
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = days_df.iloc[0]['date']

        for _, day_row in days_df.iterrows():
            day_date = day_row['date']
            day_session = day_row['session'] or "Workout"
            day_ex_count = int(day_row['exercises'])
            day_sets = int(day_row['total_sets'])
            is_selected = (day_date == st.session_state.selected_date)

            col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
            with col1:
                st.markdown(f"**{day_date}**")
            with col2:
                st.markdown(f"*{day_session.title()}*")
            with col3:
                st.markdown(f"{day_ex_count} exercises")
            with col4:
                if st.button("▼ Hide" if is_selected else "▶ View", key=f"btn_{day_date}"):
                    st.session_state.selected_date = day_date if not is_selected else None
                    st.rerun()

            if is_selected:
                day_exercises = df[df['date'] == day_date].sort_values('exercise_name')
                weighted = day_exercises[day_exercises['weight_lbs'] > 0]

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Exercises", day_ex_count)
                m2.metric("Total Sets", day_sets)
                m3.metric("Max Weight", f"{weighted['weight_lbs'].max() if not weighted.empty else 0} lbs")
                m4.metric("Heaviest Lift",
                          weighted.loc[weighted['weight_lbs'].idxmax(), 'exercise_name'] if not weighted.empty else "—")

                display_df = day_exercises[['exercise_name', 'weight_lbs', 'sets', 'reps', 'notes']].copy()
                display_df.columns = ['Exercise', 'Weight (lbs)', 'Sets', 'Reps', 'Notes']
                st.dataframe(
                    display_df.style.map(highlight_notes, subset=['Notes']),
                    use_container_width=True, hide_index=True
                )

                st.markdown("**Progressive Overload Status (exercises performed today):**")
                for ex_name in day_exercises['exercise_name'].unique():
                    ex_history = df[df['exercise_name'] == ex_name]
                    if len(ex_history) >= 2 and ex_history['weight_lbs'].max() > 0:
                        overload = analyze_progressive_overload(ex_history)
                        if overload['trend'] == 'stagnant':
                            st.warning(f"**{ex_name}:** {overload['message']}")
                        elif overload['trend'] == 'improving':
                            st.success(f"**{ex_name}:** {overload['message']}")

            st.divider()

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — LOG WORKOUT
# ───────────────────────────────────────────────────────────────────────────────
with tab_log:
    st.subheader("➕ Log a New Workout Day")
    st.caption("Exercises are parsed by the AI and added to your history automatically.")

    # Date & session
    col_date, col_session = st.columns([1, 2])
    with col_date:
        workout_date_input = st.date_input("Workout Date", value=date.today())
    with col_session:
        session_input = st.text_input(
            "Session Type",
            placeholder="e.g. chest & triceps, legs, back & biceps, cardio"
        )

    st.write("---")

    # Dynamic exercise rows
    st.markdown("**Exercises** — enter one per line in your natural shorthand:")
    st.caption("e.g. `Bench press: 135lbs, 4x10`  ·  `Pull-ups: 4x8`  ·  `Incline treadmill: 8% @ 3.6mph, 25 min`")

    if 'exercise_rows' not in st.session_state:
        st.session_state.exercise_rows = [""] * 5

    updated_rows = []
    for i, val in enumerate(st.session_state.exercise_rows):
        updated_rows.append(st.text_input(
            f"Exercise {i + 1}", value=val, key=f"ex_row_{i}",
            label_visibility="collapsed",
            placeholder=f"Exercise {i + 1}"
        ))
    st.session_state.exercise_rows = updated_rows

    col_add, col_clear, _ = st.columns([1, 1, 5])
    with col_add:
        if st.button("+ Add Row"):
            st.session_state.exercise_rows.append("")
            st.rerun()
    with col_clear:
        if st.button("Clear Rows"):
            st.session_state.exercise_rows = [""] * 5
            st.rerun()

    st.write("---")

    if st.button("💾 Save & Process Workout", type="primary"):
        exercise_lines = [l.strip() for l in st.session_state.exercise_rows if l.strip()]
        iso_date = workout_date_input.strftime("%Y-%m-%d")

        # Validate
        if not session_input.strip():
            st.error("Please enter a session type (e.g. 'chest & triceps').")
            st.stop()
        if not exercise_lines:
            st.error("Please enter at least one exercise.")
            st.stop()

        # Warn on duplicate date — don't block, user may want to add a second session
        existing_dates = df['date'].unique().tolist() if not df.empty else []
        if iso_date in existing_dates:
            st.warning(
                f"A workout for **{iso_date}** already exists. "
                f"A second entry will be appended for that date."
            )

        with st.status("Processing workout...", expanded=True) as status:

            # Step 1 — Append to .txt
            try:
                append_day_to_txt(iso_date, session_input.strip(), exercise_lines)
                st.write(f"✅ Appended to `{RAW_FILE}`")
            except Exception as e:
                st.error(f"Failed to write workout log: {e}")
                st.stop()

            # Step 2 — Ingest each line through qwen2.5
            st.write(f"🤖 Parsing {len(exercise_lines)} exercise(s) with {INGESTION_MODEL}...")

            with open(DB_PATH, 'r') as f:
                existing_records = json.load(f)

            next_day_id = max((r['day_id'] for r in existing_records), default=0) + 1
            new_records = []
            failed_lines = []

            for line in exercise_lines:
                if should_skip(line):
                    st.write(f"⏭ Skipped: `{line}`")
                    continue
                try:
                    record = ingest_exercise_line(line, next_day_id, iso_date, session_input.strip())
                    if record:
                        new_records.append(record)
                        st.write(f"✅ {record['exercise_name']}")
                except json.JSONDecodeError as e:
                    st.write(f"⚠️ Parse failed: `{line}` — {e}")
                    failed_lines.append(line)
                except Exception as e:
                    st.write(f"❌ Error: `{line}` — {e}")
                    failed_lines.append(line)

            # Step 3 — Save to exercise_db.json
            if new_records:
                existing_records.extend(new_records)
                with open(DB_PATH, 'w') as f:
                    json.dump(existing_records, f, indent=4)
                st.write(f"✅ Saved {len(new_records)} exercise(s) to `{DB_PATH}`")

            if failed_lines:
                st.warning(f"⚠️ {len(failed_lines)} line(s) could not be parsed: {failed_lines}")

            status.update(
                label=f"Done! {len(new_records)} exercises logged for {iso_date}.",
                state="complete"
            )

        st.success(f"✅ **{iso_date} — {session_input}** added to your history.")
        st.info("Switch to the History tab to see your new workout.")

        # Reset form
        st.session_state.exercise_rows = [""] * 5

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — COACH GT
# ───────────────────────────────────────────────────────────────────────────────
with tab_coach:
    selected_date = st.session_state.get('selected_date')
    if selected_date:
        days_df_ref = df.groupby(['date', 'session']).size().reset_index(name='count')
        match = days_df_ref[days_df_ref['date'] == selected_date]['session'].values
        session_label = match[0] if len(match) > 0 else "your workout"
        st.caption(
            f"📌 Reference day: **{selected_date}** — *{session_label}*. "
            f"Coach always has your full history and can answer questions about any session or exercise."
        )
    else:
        st.caption(
            "Coach always has access to your full workout history and science database. "
            "Select a day in the History tab to highlight it as a reference point."
        )

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if st.session_state.chat_messages:
        if st.button("🗑 Clear Chat", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()

    user_question = st.chat_input("Ask about your progress or training science...")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_messages.append({"role": "user", "content": user_question})

        # RAG retrieval
        with st.spinner("Searching exercise science database..."):
            db = load_science_db()
            relevant_docs = db.similarity_search(user_question, k=5)
            science_context = "\n\n".join([
                f"[Science source {i + 1}]\n"
                f"Article title: {doc.metadata.get('article_title', 'Unknown title')}\n"
                f"Content:\n{doc.page_content}"
                for i, doc in enumerate(relevant_docs)
            ])

        with st.expander("Retrieved science sources"):
            for doc in relevant_docs:
                st.write(doc.metadata)

        history_context = build_history_context(df, selected_date)

        system_prompt = f"""You are {COACH_NAME}, a high-level Strength and Conditioning Specialist.

USER PROFILE:
- Height: {user_profile['height']}
- Body Weight: {user_profile['weight_lbs']} lbs
- Age: {user_profile['age']}
- Primary Goal: {user_profile['goal']}
- Training Experience: {user_profile['experience']}
- Available Training Days Per Week: {user_profile['days_per_week']}
- Injuries / Limitations: {user_profile['injuries']}

WORKOUT DATA AND SCIENCE CONTEXT:
{history_context}

SCIENCE CONTEXT:
{science_context}

INSTRUCTIONS:
- You have the user's full workout history above. Reference it freely across all questions.
- Tailor all advice to the user's stated goal ({user_profile['goal']}) and experience level ({user_profile['experience']}).
- Always account for any listed injuries ({user_profile['injuries']}) or limitations before recommending exercises or loads.
- Use your knowledge base whenever possible and try to cite science or research in every response.
- When citing training principles and science-based facts, cite only the Article title values shown in {science_context}.
- Do not cite filenames, source paths, chunk numbers, or metadata keys.
- Do not invent article titles. If no Article title is available, say the source title is unavailable.
- Do not ever reference the NSCA's "basics of strength and conditioning manual"; only use the concepts as part of your foundational knowledge base.
- If the science context does not cover a topic, say so explicitly rather than speculating.
- Be direct and concise. Do not be overly technical. Translate technical language to common language when it makes sense. Avoid motivational filler.
- You remember everything said earlier in this conversation — refer back to it naturally when relevant."""

        # ── BACKEND ROUTING ───────────────────────────────────────────────────
        # Reads COACH_BACKEND from secrets at runtime — no code change needed
        # when deploying to Streamlit Cloud vs. running locally.
        coach_backend = st.secrets.get("COACH_BACKEND", "local")

        messages_for_model = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.chat_messages:
            messages_for_model.append({"role": msg["role"], "content": msg["content"]})

        def stream_coach_response():
            try:
                if coach_backend == "cloud":
                    # ── CLOUD: Gemini 2.5 Flash Lite ─────────────────────────
                    import google.generativeai as genai

                    try:
                        gemini_api_key = st.secrets["GEMINI_API_KEY"]
                    except KeyError:
                        yield "❌ GEMINI_API_KEY not found in secrets. Add it to `.streamlit/secrets.toml`."
                        return

                    genai.configure(api_key=gemini_api_key)
                    model = genai.GenerativeModel(
                        model_name=CLOUD_COACH_MODEL,
                        system_instruction=system_prompt,
                        generation_config={"temperature": 0.3},
                    )

                    # Convert message history to Gemini format
                    # Gemini uses "user"/"model" roles (not "user"/"assistant")
                    gemini_history = []
                    for msg in st.session_state.chat_messages[:-1]:  # All but the latest
                        gemini_history.append({
                            "role": "model" if msg["role"] == "assistant" else "user",
                            "parts": [msg["content"]]
                        })

                    chat = model.start_chat(history=gemini_history)
                    response_stream = chat.send_message(
                        st.session_state.chat_messages[-1]["content"],
                        stream=True
                    )
                    for chunk in response_stream:
                        yield chunk.text

                else:
                    # ── LOCAL: Ollama + qwen3.6 ──────────────────────────────
                    stream = ollama.chat(
                        model=LOCAL_COACH_MODEL,
                        messages=messages_for_model,
                        stream=True,
                        options={'temperature': 0.3, 'think': False}
                    )
                    for chunk in stream:
                        yield chunk['message']['content']

            except Exception as e:
                yield f"❌ {COACH_NAME} encountered an error: {e}"

        with st.chat_message("assistant"):
            response = st.write_stream(stream_coach_response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})