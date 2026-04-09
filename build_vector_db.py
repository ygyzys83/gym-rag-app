import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Paths
MD_FILE = "data/knowledge_markdown/science_base.md"
CHROMA_PATH = "data/chroma_db"


def build_db():
    if not os.path.exists(MD_FILE):
        print("Markdown file not found!")
        return

    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_content = f.read()

    # 2. Smart Chunking: It keeps the headers with the text!
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    chunks = splitter.split_text(md_content)
    print(f"Created {len(chunks)} smart chunks from the {MD_FILE} file.")

    # 3. Create Embeddings (Turning words into numbers)
    # This runs locally on your RTX 5070
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Save to ChromaDB
    print("Building Vector Database... this may take a moment.")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Success! Vector DB built and saved to {CHROMA_PATH}")


if __name__ == "__main__":
    build_db()