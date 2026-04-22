import hashlib
import os
from typing import List, Literal
from guardrails import normalize_whitespace

from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
import chromadb

from config import Settings

MAX_CONTEXT_CHARS = 11000
BREAKPOINT_THRESHOLD_AMMOUNT = 0.7
BREAKPOINT_THRESHOLD_TYPE = "gradient"



def calculate_tokens_per_query(text: str) -> int:
    return max(1, len(text)//4)



def clip_text(text:str, max_chars: int) -> str:
    return text[:max_chars] if len(text) > max_chars else text



def doc_id(doc: Document) -> str:
    
    src = str(doc.metadata.get("source", ""))
    content_hash = hashlib.md5(doc.page_content[:500].encode()).hexdigest()
    return f"{src}:{content_hash}"



def remove_duplicated_retrieved_chunks(chunks: List[Document]) -> List[Document]:

    chunks_seen = set()
    final_chunks = list()

    for single_chunk in chunks:
        chunk_id = doc_id(single_chunk)

        if chunk_id not in chunks_seen:
            chunks_seen.add(chunk_id)
            final_chunks.append(single_chunk)

    return final_chunks



def format_docs_for_context(docs: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:

    chunks = list()
    total = 0

    for i, doc in enumerate(docs, start=1):

        src = doc.metadata.get("source", "unknown")
        snippet = normalize_whitespace(doc.page_content)
        block = f"[Source {i}: {src}]\n{snippet}"

        if total + len(block) > max_chars:
            break

        chunks.append(block)
        total += len(block)
        
    return "\n\n".join(chunks)



def load_documents_from_files(file_paths: List[str]) -> List[Document]:

    docs = list()
    for single_file_path in file_paths:

        if not os.path.exists(single_file_path):
            continue

        normalized_path = single_file_path.lower()

        if normalized_path.endswith(".txt"):
            docs.extend(TextLoader(normalized_path,encoding="utf-8").load())
        
        elif normalized_path.endswith(".pdf"):
            docs.extend(PyPDFLoader(normalized_path).load())

        elif normalized_path.endswith(".md"):
            docs.extend(UnstructuredMarkdownLoader(normalized_path).load())

    return docs



def load_documents_from_urls(web_urls: List[str]) -> List[Document]:

    docs = list()
    for single_url in web_urls:
        docs.extend(WebBaseLoader(single_url).load())

    return docs



def load_all_documents(web_urls, local_files) -> List[Document]:

    all_docs = list()

    all_docs.extend(load_documents_from_urls(web_urls))
    all_docs.extend(load_documents_from_files(local_files))

    return all_docs



def build_retriever(all_documents: List[Document], embedding_model: str, 
                    breakpoint_threshold_type: Literal['percentile','standard_deviation','interquartile','gradient'], 
                    breakpoint_threshold_amount: float, min_chunk_size: int, top_k_chunks: int):

    settings = Settings.from_env()
    semantic_chunker = SemanticChunker(embeddings=OpenAIEmbeddings(model=embedding_model),
                                       breakpoint_threshold_type=breakpoint_threshold_type,
                                       breakpoint_threshold_amount=breakpoint_threshold_amount,
                                       min_chunk_size=min_chunk_size)
    
    chunk_documents = semantic_chunker.split_documents(all_documents)

    estimated_embed_tokens = sum(calculate_tokens_per_query(d.page_content) for d in chunk_documents)

    chromadb_client = chromadb.HttpClient(host="localhost",port=8000)
    collection = chromadb_client.get_or_create_collection(name="research-docs")

    vector_store = Chroma(collection_name="research-docs",
                          client=chromadb_client,
                          embedding_function=OpenAIEmbeddings(model=embedding_model))

    if len(all_documents) > 0 or all_documents != None:
        vector_store.add_documents(chunk_documents)
    
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k_chunks})

    return retriever, chunk_documents, estimated_embed_tokens
