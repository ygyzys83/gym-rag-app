Note: This is a Local-First AI application designed to run on personal hardware (RTX 5070) for maximum data privacy.

# 🏋️‍♂️ Elite AI Fitness Tracker & RAG Coach

A privacy-first, local-AI fitness dashboard that transforms messy workout journals into structured analytics and provides a science-backed "AI Coach" using RAG (Retrieval-Augmented Generation).

## 🚀 The Technical Challenge
Workout data is notoriously messy. Converting human shorthand (e.g., "70s for 3x10") into a database often results in "Lazy AI" errors where models summarize or skip entries. This project implements a **Brute-Force Ingestion Pipeline** and a **Multi-Model RAG Stack** to solve these challenges with 100% data fidelity.

## 🏗 System Architecture

```mermaid
graph TD
    subgraph Ingestion
        A[.txt Journal] -->|Line-by-Line| B[Llama 3.2]
        B -->|Structured| C[(JSON Database)]
    end

    subgraph Knowledge Base
        D[NSCA Science PDF] -->|Docling| E[Markdown]
        E -->|Smart Chunking| F[(ChromaDB Vector Store)]
    end

    subgraph User UI
        G[Streamlit Dashboard] -->|Query| H[RAG Logic]
        C --> H
        F --> H
        H -->|Context + History| I[Coach Arnold - Llama 3.2]
        I -->|Streaming Advice| G
    end

## 🛠 Tech Stack
- **Languages:** Python (Pandas, Re, Asyncio)
- **AI Models:** Llama 3.2 (Extraction & Coaching), Qwen 3.5 (Reasoning)
- **Inference Engine:** Ollama (Local GPU acceleration via RTX 5070)
- **PDF Processing:** Docling (IBM’s Layout-Aware Parser)
- **Vector Database:** ChromaDB
- **Framework:** Streamlit

## 🌟 Key Features
- **100% Reliable Ingestion:** Uses a custom brute-force line-by-line parsing strategy to ensure no workout is skipped.
- **Science-Backed Advice:** The "AI Coach" retrieves actual training principles from a 100-page NSCA manual before answering.
- **Interactive Performance Matrix:** A dynamic UI that tracks volume, max weight, and highlights "PR" (Personal Records) using sentiment analysis.
- **Privacy First:** 100% local. No data ever leaves your hardware.

## 🛠 Installation
1. Install [Ollama](https://ollama.com) and pull models: `ollama pull llama3.2`
2. Clone this repo.
3. Place your training manuals (PDF) into the knowledge_base/ folder.
4. Install dependencies: `pip install -r requirements.txt`
5. Run ingestion: `python ingest_data.py`
6. Launch UI: `streamlit run main_app.py`