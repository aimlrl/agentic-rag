import os
from dataclasses import dataclass
from typing import Optional, Literal, cast
from dotenv import load_dotenv

BreakpointThresholdType = Literal["percentile", "standard_deviation", "interquartile", "gradient"]


@dataclass
class Settings:
    openai_api_key: str
    rag_response_llm: str = "gpt-4o-mini"
    grader_llm: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    user_agent: str = "agentic-rag-app/1.0"
    app_env: str = "preproduction"
    langsmith_api_key: Optional[str] = None
    langsmith_tracing: bool = False
    min_chunk_size: int = 50
    top_k_chunks: int = 6
    top_k_after_filter_chunks: int = 4
    breakpoint_threshold_type: BreakpointThresholdType = "gradient"
    breakpoint_threshold_amount: float = 0.7
    max_context_chars: int = 11000
    max_query_chars: int = 2000
    min_query_chars: int = 3
    max_rewrite_tries: int = 2
    min_retrieved_chunks: int = 2
    min_relevant_chunks: int = 1
    min_context_relevancy: float = 0.7
    min_faithfullness: float = 0.9
    max_response_sentences: int = 8
    embedding_cost_per_1m_tokens: float = 0.02
    input_tokens_cost_per_1m_tokens: float = 2.00
    output_tokens_cost_per_1m_tokens: float = 8.00

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        threshold_type = os.getenv("BREAKPOINT_THRESHOLD_TYPE", "gradient")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")

        return cls(
            openai_api_key=openai_api_key,
            rag_response_llm=os.getenv("RAG_RESPONSE_LLM", "gpt-4o-mini"),
            grader_llm=os.getenv("GRADER_LLM", os.getenv("RAG_RESPONSE_LLM", "gpt-4o-mini")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            user_agent=os.getenv("USER_AGENT", "agentic-rag-app/1.0"),
            app_env=os.getenv("APP_ENV", "preproduction"),
            langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
            langsmith_tracing=os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
            min_chunk_size=int(os.getenv("CHUNK_SIZE", 50)),
            top_k_chunks=int(os.getenv("TOP_K_CHUNKS", 6)),
            top_k_after_filter_chunks=int(os.getenv("TOP_K_AFTER_FILTER_CHUNKS", 4)),
            breakpoint_threshold_type=cast(BreakpointThresholdType, threshold_type),
            breakpoint_threshold_amount=float(os.getenv("BREAKPOINT_THRESHOLD_AMOUNT", 0.7)),
            max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", 11000)),
            max_query_chars=int(os.getenv("MAX_QUERY_CHARS", 2000)),
            min_query_chars=int(os.getenv("MIN_QUERY_CHARS", 3)),
            max_rewrite_tries=int(os.getenv("MAX_REWRITE_TRIES", 2)),
            min_retrieved_chunks=int(os.getenv("MIN_RETRIEVED_CHUNKS", 2)),
            min_relevant_chunks=int(os.getenv("MIN_RELEVANT_CHUNKS", 1)),
            min_context_relevancy=float(os.getenv("MIN_CONTEXT_RELEVANCY", 0.7)),
            min_faithfullness=float(os.getenv("MIN_FAITHFULLNESS", 0.9)),
            max_response_sentences=int(os.getenv("MAX_RESPONSE_SENTENCES", 8)),
            embedding_cost_per_1m_tokens=float(os.getenv("EMBEDDING_COST_PER_1M_TOKENS", 0.02)),
            input_tokens_cost_per_1m_tokens=float(os.getenv("INPUT_TOKENS_COST_PER_1M_TOKENS", 2.00)),
            output_tokens_cost_per_1m_tokens=float(os.getenv("OUTPUT_TOKENS_COST_PER_1M_TOKENS", 8.00)))
