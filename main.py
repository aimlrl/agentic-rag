import json
import os

from config import Settings
from ingestion import load_all_documents, build_retriever
from graph import AgenticRAGGraph


def bootstrap_sources():
    web_urls = [
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    ]
    local_files = []
    return web_urls, local_files


def main():
    settings = Settings.from_env()
    os.environ["USER_AGENT"] = settings.user_agent
    web_urls, local_files = bootstrap_sources()
    documents = load_all_documents(web_urls, local_files)
    retriever, chunk_documents, estimated_embedding_tokens = build_retriever(documents, settings.embedding_model,
                                                                    settings.breakpoint_threshold_type,
                                                                    settings.breakpoint_threshold_amount,
                                                                    settings.min_chunk_size,
                                                                    settings.top_k_chunks)
    
    app = AgenticRAGGraph(settings, retriever, estimated_embedding_tokens)

    print(f"Indexed {len(chunk_documents)} chunks from {len(documents)} documents.")
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        result = app.ask_agent(query)
        print("\nANSWER:\n", result["answer"])
        print("\nMETRICS:\n", json.dumps(result["metrics"], indent=2))


def demo_eval():
    settings = Settings.from_env()
    os.environ["USER_AGENT"] = settings.user_agent
    web_urls, local_files = bootstrap_sources()
    documents = load_all_documents(web_urls, local_files)
    retriever, chunk_documents, estimated_embedding_tokens = build_retriever(documents, settings.embedding_model,
                                                                    settings.breakpoint_threshold_type,
                                                                    settings.breakpoint_threshold_amount,
                                                                    settings.min_chunk_size,
                                                                    settings.top_k_chunks)
    app = AgenticRAGGraph(settings, retriever, estimated_embedding_tokens)


if __name__ == "__main__":
    main()
