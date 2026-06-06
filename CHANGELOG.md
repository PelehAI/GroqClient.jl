# Changelog

## 0.1.0 (unreleased)

- Initial release. Minimal Julia client for Groq's OpenAI-compatible Chat
  Completions API.
- `chat` / `chat_async` against `/openai/v1/chat/completions` with HTTP
  keep-alive pooling.
- Per-client sliding-window RPM semaphore shared across concurrent calls.
- Per-reply token + USD cost accounting (uncached input, cached reads, output)
  against a bundled per-model price table (`openai/gpt-oss-20b`).
- `reasoning_effort` passthrough (defaults to `"low"`); the reasoning channel
  is dropped from `Reply.text` (final answer only).
- `response_format` passthrough for JSON Object / JSON Schema structured output.
- `Budget` wrapper that throws `BudgetExceeded` on cap.
- `retry-after`-aware 429 handling; bounded exponential backoff on 5xx.
- Stub-friendly: body-building and reply-parsing are pure functions, so tests
  run with no network and no API key.
- Sibling to `AnthropicClient.jl` — same public surface and `Reply` layout.
