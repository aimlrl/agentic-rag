import hashlib
import time
from typing import List, Optional
from dataclasses import dataclass, field
from config import Settings



@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class RunTimeMetrics:

    request_id: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    retrieval_count: int = 0
    relevant_retrieval_count: int = 0
    query_rewrite_count: int = 0
    tool_calls: int = 0
    trajectory: List[str] = field(default_factory=list)
    llm_calls: int = 0
    grader_calls: int = 0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    embedding_token_usage: int = 0
    input_guardrail_triggered: bool = False
    injection_detected: bool = False
    response_refusal: bool = False
    is_hallucination : bool = False
    final_factualness_score: float = 0.0
    final_retrieval_relevance_score: float = 0.0
    final_answer_relevance_score: float = 0.0
    final_confidence_score: float = 0.0
    error: Optional[str] = None

    @property
    def latency_in_seconds(self) -> float:
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def estimated_chat_cost_usd(self) -> float:

        settings = Settings.from_env()
        input_cost = (self.token_usage.input_tokens / 1000000) * settings.input_tokens_cost_per_1m_tokens
        output_cost = (self.token_usage.output_tokens / 1000000) * settings.output_tokens_cost_per_1m_tokens
        
        return input_cost + output_cost
    
    @property
    def estimated_embedding_cost_usd(self) -> float:

        settings = Settings.from_env()
        return (self.embedding_token_usage / 1000000) * settings.embedding_cost_per_1m_tokens
    
    @property
    def estimated_total_cost_usd(self) -> float:
        return self.estimated_chat_cost_usd + self.estimated_embedding_cost_usd



def create_request_id(text: str) -> str:
    return hashlib.md5(f"{text}-{time.time()}".encode()).hexdigest()


def get_tokens_usage(response) -> TokenUsage:

    usage = getattr(response, "usage_metadata", None) or {}
    return TokenUsage(input_tokens = int(usage.get("input_tokens", 0) or 0),
                      output_tokens = int(usage.get("output_tokens",0) or 0),
                      total_tokens = int(usage.get("total_tokens",0) or 0))


def compute_total_tokens_comsumption(metrics: RunTimeMetrics, tokens_usage: TokenUsage):

    metrics.token_usage.input_tokens += tokens_usage.input_tokens
    metrics.token_usage.output_tokens += tokens_usage.output_tokens
    metrics.token_usage.total_tokens += tokens_usage.total_tokens
