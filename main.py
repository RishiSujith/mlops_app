import argparse
from indexer import index_tree
from graph import build_graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",    default="infinity_war.json")
    parser.add_argument("--reindex", action="store_true")
    args = parser.parse_args()

    collection = index_tree(args.json, force_reindex=args.reindex)
    graph      = build_graph(collection)

    print("🌌 Ready! Type 'q' to quit.\n")

    while True:
        question = input("❓ Ask: ").strip()
        if not question or question.lower() in ("q", "quit", "exit"):
            break

        result = graph.invoke({"question": question, "documents": [], "generation": ""})
        print(f"\n💡 {result['generation']}\n{'-'*50}\n")


if __name__ == "__main__":
    main()