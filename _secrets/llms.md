# LLM Backends — MultiModalRag

## Startup backend selection

| Priority | Condition                        | Backend          | Model                              |
| -------- | -------------------------------- | ---------------- | ---------------------------------- |
| 1        | `GROQ_API_KEY` set               | **Groq**         | `llama-3.3-70b-versatile`          |
| 2        | `USE_HF_LLM=1` + `HF_TOKEN` set  | **HF Inference** | `meta-llama/Llama-3.1-8B-Instruct` |
| 3        | Otherwise                        | **Ollama**       | `llama3.2`                         |

## Runtime fallback chain (Groq backend)

When Groq hits its daily/per-minute rate limit, the engine cascades automatically:

```
1. llama-3.3-70b-versatile  (primary — tighter daily quota)
      ↓ rate-limited
2. llama-3.1-8b-instant     (Groq fallback — separate, larger daily quota)
      ↓ also rate-limited
3. HF Inference API         (if AgenticMultiModalRag_Token is set on Space)
      ↓ credits depleted / unavailable
4. Ollama                   (local only — not running on HF Space)
      ↓ not reachable
5. Error message shown to user
```

Applies to both `RAGEngine._query_groq` and `LLMTool._call_groq`.

## Local dev

```text
BACKEND = "groq"   (if GROQ_API_KEY is loaded from _secrets/)
BACKEND = "ollama" (fallback if no API keys found)
Requires: _secrets/*.txt with gsk_... line, or Ollama running
```

## HuggingFace Space

```text
BACKEND = "groq"
Primary model: llama-3.3-70b-versatile
Fallback model: llama-3.1-8b-instant (automatic, no config needed)
Requires: GROQ_API_KEY set in Space → Settings → Secrets
Get key: https://console.groq.com
```
