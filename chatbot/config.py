import os


GROQ_MODEL        = "llama-3.3-70b-versatile"

EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"   # swap freely, ~130MB, CPU-friendly

CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "infinity_war_rag"

TOP_K = 3