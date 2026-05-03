import os
import shutil
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import re

# ── CONFIG ────────────────────────────────────────────────────────────────────
MD_DIR = "data/knowledge_markdown"  # Indexes ALL .md files in this folder
CHROMA_PATH = "data/chroma_db"

# Two-pass chunking: header splitter first, then cap oversized chunks
CHUNK_SIZE = 1000  # Max characters per chunk
CHUNK_OVERLAP = 150  # Overlap between chunks to preserve context across boundaries


def extract_article_title(content: str, filename: str) -> str:
    """Use the first H1 as the article title; fallback to filename."""
    match = re.search(r"^#\s+(.+?)\s*$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return os.path.splitext(filename)[0]

def load_markdown_files(md_dir: str) -> list[tuple[str, str]]:
    """Return a list of (filename, content) for every .md file in md_dir."""
    if not os.path.exists(md_dir):
        print(f"❌ Markdown directory not found: {md_dir}")
        return []

    files = [f for f in os.listdir(md_dir) if f.endswith(".md")]
    if not files:
        print(f"❌ No .md files found in {md_dir}. Run ingest_knowledge.py first.")
        return []

    results = []
    for fname in sorted(files):
        path = os.path.join(md_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            results.append((fname, f.read()))
        print(f"  Loaded: {fname}")
    return results


def chunk_markdown(filename: str, content: str) -> list:
    """
    Two-pass chunking strategy:
    Pass 1 — MarkdownHeaderTextSplitter: splits on headers, keeping context.
    Pass 2 — RecursiveCharacterTextSplitter: further splits any chunks that are
              still too large.
    """
    article_title = extract_article_title(content, filename)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
    )
    header_chunks = header_splitter.split_text(content)

    for chunk in header_chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["article_title"] = article_title

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    final_chunks = char_splitter.split_documents(header_chunks)

    return final_chunks


def build_db():
    print("=" * 50)
    print("Building Vector Database")
    print("=" * 50)

    # Load all markdown files
    print(f"\n📂 Scanning {MD_DIR} for markdown files...")
    md_files = load_markdown_files(MD_DIR)
    if not md_files:
        return

    # Chunk all files
    all_chunks = []
    for filename, content in md_files:
        chunks = chunk_markdown(filename, content)
        all_chunks.extend(chunks)
        char_counts = [len(c.page_content) for c in chunks]
        print(f"  {filename}: {len(chunks)} chunks "
              f"(avg {sum(char_counts) // len(char_counts)} chars, "
              f"max {max(char_counts)} chars)")

    print(f"\n📦 Total chunks across all files: {len(all_chunks)}")

    # ── DEDUP PROTECTION ──────────────────────────────────────────────────────
    # Chroma.from_documents APPENDS to an existing DB, causing duplicates on re-runs.
    # We wipe and rebuild cleanly every time.
    if os.path.exists(CHROMA_PATH):
        print(f"\n🗑  Existing DB found at {CHROMA_PATH} — wiping for clean rebuild.")
        shutil.rmtree(CHROMA_PATH)

    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Build embeddings and save
    print("\n🔢 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("💾 Building and saving Vector DB (this may take a moment)...")
    Chroma.from_documents(
        all_chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"\n✅ Vector DB built and saved to {CHROMA_PATH}")
    print(f"   {len(all_chunks)} chunks from {len(md_files)} source file(s) indexed.")
    print("   Run test_search.py to verify retrieval quality.")


if __name__ == "__main__":
    build_db()