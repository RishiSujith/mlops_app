from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import GROQ_MODEL, TOP_K
from dotenv import load_dotenv
load_dotenv()
import os
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

class RAGState(TypedDict):
    question  : str
    documents : list[dict]
    generation: str


def build_graph(collection):
    llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

    def retrieve(state: RAGState) -> RAGState:
        results = collection.query(
            query_texts=[state["question"]],
            n_results=TOP_K,
            include=["documents", "metadatas"],
        )
        docs = [
            {"content": doc, "title": meta["title"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]
        return {**state, "documents": docs}

    def generate(state: RAGState) -> RAGState:
        context = "\n\n---\n\n".join(d["content"] for d in state["documents"])
        response = llm.invoke([
            SystemMessage(content=(
                "You are an expert on the MCU. Answer using only the context below. "
                "Be specific, name characters, locations, and events."
                "Keep your answers simple and short unless directed to give detailed output"
            )),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['question']}"),
        ])
        return {**state, "generation": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()