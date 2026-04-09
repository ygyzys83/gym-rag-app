from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import ollama
import streamlit as st
import pandas as pd
import json
import os

@st.cache_resource
def load_science_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
    return db

st.set_page_config(page_title="Elite Fitness Tracker", layout="wide")
st.title("🏋️‍♂️ Workout Performance Matrix")

DB_PATH = 'data/processed/exercise_db.json'

if os.path.exists(DB_PATH):
    with open(DB_PATH, 'r') as f:
        df = pd.DataFrame(json.load(f))

    # Sidebar
    st.sidebar.header("Filter Options")
    exercise_list = sorted(df['exercise_name'].unique())
    selected_ex = st.sidebar.selectbox("Select Exercise:", exercise_list)

    # Sidebar --- USER PROFILE SECTION
    st.sidebar.write("---")
    st.sidebar.header("👤 Your Profile")

    # We use st.number_input for precision and st.text_input for height (e.g. 5'11")
    user_height = st.sidebar.text_input("Height (e.g., 6'0\")", value="6'0\"")
    user_weight = st.sidebar.number_input("Weight (lbs)", value=190, step=1)
    user_age = st.sidebar.number_input("Age", value=43, step=1)

    # Save profile to a local variable for the coach to use
    user_profile = {
        "height": user_height,
        "weight": user_weight,
        "age": user_age
    }

    # Filter data
    view_data = df[df['exercise_name'] == selected_ex].sort_values('day_id', ascending=False)

    # 1. Performance Matrix (Top Row)
    st.subheader(f"Performance Matrix: {selected_ex}")
    m1, m2, m3, m4 = st.columns(4)

    if not view_data.empty:
        # Latest stats
        latest = view_data.iloc[0]
        m1.metric("Last Weight", f"{latest['weight_lbs']} lbs")
        m2.metric("Last Volume", f"{latest['sets']} x {latest['reps']}")

        # All-time stats
        m3.metric("Max Weight", f"{view_data['weight_lbs'].max()} lbs")
        m4.metric("Total Sets Done", view_data['sets'].sum())

        # 2. Detailed History Matrix
        st.write("---")
        st.subheader("Workout History (Newest First)")

        # We rename columns to look professional
        matrix_df = view_data[['day_id', 'weight_lbs', 'sets', 'reps', 'notes']].copy()
        matrix_df.columns = ['Day #', 'Weight (lbs)', 'Sets', 'Reps', 'Notes']

        # Professional touch: Highlight PRs or positive notes in the table
        def highlight_notes(val):
            note_text = str(val).upper()
            # We add the specific keywords found in your actual JSON
            wins = ['STRONG', 'HEAVY', 'WEIGHTED', 'PR', 'PR!']

            if any(word in note_text for word in wins):
                return 'background-color: #004d00; color: white'  # Dark green
            return ''

        # Display the matrix as an interactive table
        st.dataframe(
            matrix_df.style.map(highlight_notes, subset=['Notes']),
            use_container_width=True,
            hide_index=True
        )

        # --- AI COACH SECTION ---
        st.write("---")
        st.subheader("💬 Ask Coach GT (Science-Backed)")

        # Use a form to capture the 'Enter' key press
        with st.form("coach_form", clear_on_submit=False):
            user_question = st.text_input(
                "Ask about your progress or general training science:",
                placeholder="e.g., Based on the manual, is my current volume optimal for hypertrophy?"
            )
            submit_button = st.form_submit_button("Ask Coach")

        if submit_button and user_question:
            # 1. RETRIEVE SCIENCE (The RAG Part)
            with st.spinner("Searching exercise manuals..."):
                db = load_science_db()  # Instant after first load
                relevant_docs = db.similarity_search(user_question, k=3)
                science_context = "\n".join([doc.page_content for doc in relevant_docs])

            # 2. RETRIEVE HISTORY (The Personal Part)
            # Sending full history for cross-exercise analysis
            history_context = df.to_json(orient='records')


            # 3. GENERATE ADVICE
            def stream_coach_response():
                try:
                    stream = ollama.chat(
                        model="llama3.2",
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are a high-level Strength and Conditioning Specialist. 
                                            USER PROFILE: Height: {user_profile['height']}, Weight: {user_profile['weight']}lbs, Age: {user_profile['age']}.
                                            Use this profile and the provided Science/History to give objective, technical advice. 
                                            Avoid fluff and nicknames. Cite the 'NSCA Strength & Conditioning Manual' for technical standards."""
                            },
                            {
                                "role": "user",
                                "content": f"SCIENCE: {science_context}\n\nHISTORY: {history_context}\n\nQUESTION: {user_question}"
                            }
                        ],
                        stream=True,
                        options={'temperature': 0.3, 'think': False}
                    )
                    for chunk in stream:
                        yield chunk['message']['content']
                except Exception as e:
                    yield f"Error: {e}"

            st.write_stream(stream_coach_response)

        elif submit_button and not user_question:
            st.warning("Please type a question first!")

    else:
        # This matches the very first 'if os.path.exists(DB_PATH):'
        st.error("Database not found!")

