from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "data/chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Test query
query = "What are the recommended rep ranges for hypertrophy?"
docs = db.similarity_search(query, k=2)

print(f"\n🔍 Search results for: '{query}'")
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata}")
    print(f"Content: {doc.page_content[:300]}...")