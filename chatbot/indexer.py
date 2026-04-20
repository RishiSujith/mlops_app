import json
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL

LEVEL_LABELS = {0: "Root", 1: "Location", 2: "Event"}


def _build_chunk(node: dict, node_map: dict) -> str:
    """Prepend ancestor titles so every chunk carries its tree context."""
    ancestors, cur = [], node
    while cur["parent_id"]:
        cur = node_map[cur["parent_id"]]
        ancestors.append(cur)

    lines = []
    for anc in reversed(ancestors):
        lines.append(f"[{LEVEL_LABELS.get(anc['level'], 'Node')}] {anc['title']}")
    lines.append(f"\n[{LEVEL_LABELS.get(node['level'], 'Node')}] {node['title']}")
    lines.append(node["text_content"])
    return "\n".join(lines)


def get_collection():
    ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def index_tree(json_path: str, force_reindex: bool = False):
    collection = get_collection()

    if collection.count() > 0 and not force_reindex:
        print(f"✅ Already indexed ({collection.count()} nodes) — skipping.\n")
        return collection

    with open(json_path) as f:
        nodes = json.load(f)

    node_map = {n["node_id"]: n for n in nodes}

    ids, docs, metas = [], [], []
    for node in nodes:
        ids.append(node["node_id"])
        docs.append(_build_chunk(node, node_map))
        metas.append({"level": node["level"], "title": node["title"]})

    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"✅ Indexed {len(nodes)} nodes into ChromaDB.\n")
    return collection