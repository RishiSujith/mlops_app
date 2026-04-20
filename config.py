import os

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "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")
GROQ_MODEL        = "llama-3.3-70b-versatile"

EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"   # swap freely, ~130MB, CPU-friendly

CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "infinity_war_rag"

TOP_K = 3