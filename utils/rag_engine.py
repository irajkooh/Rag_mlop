"""
RAG engine: retrieves relevant context from vector store,
builds a strict prompt, and queries the LLM.

Backend priority:
1. GROQ_API_KEY set → Groq API (fast, 100K tokens/day free tier)
2. USE_HF_LLM=1 + HF_TOKEN set → HuggingFace Inference API
3. Otherwise → Ollama (local)
"""
import os
import re
import logging
from typing import List, Dict, Any, Generator

from utils.vector_store import VectorStoreManager
from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL  = "llama3.2"
#DEFAULT_GROQ_MODEL    = "llama-3.3-70b-versatile"
DEFAULT_GROQ_MODEL    = "openai/gpt-oss-20b"
GROQ_FALLBACK_MODEL   = "llama-3.1-8b-instant"   # separate daily quota, higher limits
DEFAULT_HF_MODEL      = "meta-llama/Llama-3.1-8B-Instruct"

HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("AgenticMultiModalRag_Token", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
USE_HF_LLM   = os.environ.get("USE_HF_LLM", "").lower() in ("1", "true", "yes")

# Pick backend: Groq first (fast), then HF Inference (only if USE_HF_LLM=1), then Ollama
if GROQ_API_KEY:
    BACKEND = "groq"
elif USE_HF_LLM and HF_TOKEN:
    BACKEND = "hf"
else:
    BACKEND = "ollama"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM_PROMPT = """You are a document assistant. Answer questions using ONLY the [CONTEXT] provided.
Rules:
1. If the context contains information relevant to the question, answer from it — even if only partially relevant.
2. Combine information from multiple context chunks if needed.
3. Only say "I DON'T KNOW" if the context truly contains NO relevant information at all.
4. Be concise. Cite source and page when available.
5. Do NOT make up information that is not in the context.
6. When answering questions about tables or structured data, apply ALL filter conditions from the question. Only include rows that match every condition — do not display or reference rows that do not match.
7. Give ONLY the final answer. Do NOT show reasoning steps, intermediate calculations, excluded rows, or any explanation of how you arrived at the answer.
8. Format key values, measurements, thresholds, and important requirements in **bold** (markdown). Use plain text for surrounding prose.
9. OCR-scanned documents may contain garbled characters in numbers and ranges (e.g., "50H,000" means "501-1,000"; "O" may mean "0"; "l" may mean "1"). Use numeric context and table structure to interpret such artifacts — do NOT use OCR garbling as a reason to say "I DON'T KNOW".
10. When the question asks about a specific person, only answer from context that explicitly mentions that person by name. Do NOT apply one person's details (experience, role, background) to another person. If the context does not contain information about the requested person, say "I DON'T KNOW".
"""

GENERAL_PROMPT = """You are a helpful AI assistant. Answer directly and concisely — final answer only, no reasoning steps.
If you don't know the answer, say so honestly.
"""

# Cosine distance threshold (0=identical, 2=opposite).
RELEVANCE_THRESHOLD = 1.2


def _make_hf_client():
    from huggingface_hub import InferenceClient
    return InferenceClient(token=HF_TOKEN)


def _make_groq_client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _make_ollama_client():
    import ollama
    return ollama.Client(host=OLLAMA_HOST)


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate_limit" in msg or "rate limit" in msg


class RAGEngine:
    def __init__(self, vector_store: VectorStoreManager, model: str = None):
        self.vs = vector_store
        if BACKEND == "hf":
            self.model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
            self._client = _make_hf_client()
            logger.info(f"LLM backend: HuggingFace Inference ({self.model})")
        elif BACKEND == "groq":
            self.model = os.environ.get("GROQ_MODEL", DEFAULT_GROQ_MODEL)
            self._client = _make_groq_client()
            logger.info(f"LLM backend: Groq ({self.model})")
        else:
            self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            self._client = _make_ollama_client()
            logger.info(f"LLM backend: Ollama ({self.model})")

    @staticmethod
    def _fix_ocr(text: str) -> str:
        """Fix common OCR garbling in scanned building-code PDFs before sending to LLM."""
        # "50H,000" → "501-1,000"  (range rows in IBC Table 1006.3.3 and similar tables)
        text = re.sub(r'(\d+)H,(\d{3})', lambda m: f"{m.group(1)}1-1,{m.group(2)}", text)
        # "81/z" / "91/z" / "81/i" / "91/i" → "8½" / "9½"  (PDF fraction "½" OCR'd as "1/z" or "1/i")
        text = re.sub(r'(\d+)1/[zi]', lambda m: f"{m.group(1)}½", text)
        return text

    @staticmethod
    def _is_question_only(text: str) -> bool:
        """True when a chunk is mostly a list of questions with no answer content."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        questions = sum(1 for l in lines if l.endswith("?"))
        return questions / len(lines) >= 0.6

    @staticmethod
    def _is_non_content_chunk(text: str) -> bool:
        """True when a chunk is noise: URL lists, error logs, or markdown headers with no prose."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return True
        url_or_header = sum(
            1 for l in lines
            if l.startswith("http") or l.startswith("chrome-extension://")
            or (l.startswith("#") and len(l.split()) <= 6)
        )
        if url_or_header / len(lines) >= 0.5:
            return True
        # Error/log lines (common patterns from Building_code samples.txt noise)
        noise_keywords = ("402 client error", "payment required", "depleted", "rate limit", "token limit")
        lowered = text.lower()
        if any(kw in lowered for kw in noise_keywords):
            return True
        return False

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "No relevant documents found."
        parts = []
        idx = 1
        for r in results:
            if self._is_question_only(r["text"]):
                continue
            if self._is_non_content_chunk(r["text"]):
                continue
            meta = r["metadata"]
            source   = meta.get("source", "unknown")
            page     = meta.get("page", "")
            doc_type = meta.get("type", "text")
            page_str = f", Page {page}" if page else ""
            type_str = f" [{doc_type}]" if doc_type != "text" else ""
            parts.append(f"[Doc {idx} — {source}{page_str}{type_str}]\n{self._fix_ocr(r['text'])}")
            idx += 1
        return "\n\n---\n\n".join(parts) if parts else "No relevant documents found."

    def _build_messages(self, question: str, context: str, memory: ConversationMemory, extra_context: str = ""):
        if extra_context:
            extra_section = (
                f"[STRUCTURED DATA from tables]\n{extra_context}\n\n"
                "Merge the structured data above with the document context below into one clean, readable answer. "
                "Bold key values and measurements.\n\n"
            )
        else:
            extra_section = ""
        user_message = (
            f"{extra_section}[CONTEXT from documents]\n{context}\n\n"
            f"[QUESTION]\n{question}\n\n"
            "Remember: Answer from the structured data and context above. If not found, say \"I DON'T KNOW\"."
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *memory.get_history_for_prompt(),
            {"role": "user", "content": user_message},
        ]

    def _is_off_topic(self, results: List[Dict[str, Any]]) -> bool:
        if not results:
            return True
        return all(r.get("distance", 1.0) > RELEVANCE_THRESHOLD for r in results)

    def _build_general_messages(self, question: str, memory: ConversationMemory):
        return [
            {"role": "system", "content": GENERAL_PROMPT},
            *memory.get_history_for_prompt(),
            {"role": "user", "content": question},
        ]

    def query(
        self,
        question: str,
        memory: ConversationMemory,
        n_results: int = 8,
        temperature: float = 0.0,
        stream: bool = False,
        source_filter: list = None,
        pre_fetched_results: list = None,
        extra_context: str = "",
    ) -> Generator[str, None, None]:
        results = pre_fetched_results if pre_fetched_results is not None else self.vs.query(question, n_results=n_results, source_filter=source_filter)

        if self._is_off_topic(results) and not extra_context:
            logger.info(f"Off-topic query (no relevant chunks): '{question[:60]}'")
            messages = self._build_general_messages(question, memory)
        else:
            context  = self._build_context(results)
            messages = self._build_messages(question, context, memory, extra_context)

        try:
            if BACKEND == "hf":
                yield from self._query_hf(messages, memory, question, temperature, stream)
            elif BACKEND == "groq":
                yield from self._query_groq(messages, memory, question, temperature, stream)
            else:
                yield from self._query_ollama(messages, memory, question, temperature, stream)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg

    def _query_hf(self, messages, memory, question, temperature, stream):
        try:
            resp = self._client.chat_completion(
                model=self.model,
                messages=messages,
                temperature=max(temperature, 0.01),
                max_tokens=2048,
            )
            answer = resp.choices[0].message.content
            memory.add("user", question)
            memory.add("assistant", answer)
            yield answer
        except Exception as e:
            if "402" in str(e) or "payment" in str(e).lower() or "depleted" in str(e).lower():
                logger.warning("HF Inference credits depleted — falling back to Ollama")
                try:
                    yield from self._ollama_fallback(messages, memory, question, temperature)
                except Exception:
                    yield "⚠️ HF Inference credits depleted and Ollama is not available. Please set a GROQ_API_KEY."
            else:
                raise

    def _ollama_fallback(self, messages, memory, question, temperature):
        import ollama
        model = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
        answer = response["message"]["content"]
        memory.add("user", question)
        memory.add("assistant", answer)
        yield answer

    def _query_groq(self, messages, memory, question, temperature, stream):
        try:
            if stream:
                response_text = ""
                with self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                ) as stream_resp:
                    for chunk in stream_resp:
                        token = chunk.choices[0].delta.content or ""
                        response_text += token
                        yield token
                memory.add("user", question)
                memory.add("assistant", response_text)
            else:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                answer = resp.choices[0].message.content
                memory.add("user", question)
                memory.add("assistant", answer)
                yield answer
        except Exception as e:
            if _is_rate_limit(e):
                # 1. Try a smaller Groq model — it has a separate, larger daily quota
                if self.model != GROQ_FALLBACK_MODEL:
                    try:
                        logger.warning(f"Groq rate limit on {self.model} — trying {GROQ_FALLBACK_MODEL}")
                        resp = self._client.chat.completions.create(
                            model=GROQ_FALLBACK_MODEL,
                            messages=messages,
                            temperature=temperature,
                        )
                        answer = resp.choices[0].message.content
                        memory.add("user", question)
                        memory.add("assistant", answer)
                        yield answer
                        return
                    except Exception as fb_exc:
                        logger.warning(f"Groq {GROQ_FALLBACK_MODEL} also failed: {fb_exc}")
                # 2. HF Inference
                if HF_TOKEN:
                    try:
                        logger.warning("Trying HF Inference fallback")
                        hf_client = _make_hf_client()
                        hf_model = os.environ.get("HF_MODEL", DEFAULT_HF_MODEL)
                        resp = hf_client.chat_completion(
                            model=hf_model,
                            messages=messages,
                            temperature=max(temperature, 0.01),
                            max_tokens=2048,
                        )
                        answer = resp.choices[0].message.content
                        memory.add("user", question)
                        memory.add("assistant", answer)
                        yield answer
                        return
                    except Exception as hf_exc:
                        logger.warning(f"HF Inference fallback failed: {hf_exc}")
                # 3. Ollama
                try:
                    yield from self._ollama_fallback(messages, memory, question, temperature)
                except Exception:
                    yield "⚠️ All LLM backends are currently unavailable (Groq daily limit reached, Ollama not running). Please try again later or upgrade at https://console.groq.com/settings/billing"
            else:
                raise

    def _query_ollama(self, messages, memory, question, temperature, stream):
        if stream:
            response_text = ""
            stream_resp = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature},
            )
            for chunk in stream_resp:
                token = chunk["message"]["content"]
                response_text += token
                yield token
            memory.add("user", question)
            memory.add("assistant", response_text)
        else:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature},
            )
            answer = response["message"]["content"]
            memory.add("user", question)
            memory.add("assistant", answer)
            yield answer

    def list_available_models(self) -> List[str]:
        return [self.model]
