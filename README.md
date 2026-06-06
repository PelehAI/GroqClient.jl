# GroqClient.jl

[![CI](https://github.com/PelehAI/GroqClient.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PelehAI/GroqClient.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Julia client for [Groq's](https://groq.com) OpenAI-compatible Chat Completions
API. Groq-only, by design. A sibling to
[`AnthropicClient.jl`](https://github.com/PelehAI/AnthropicClient.jl) — same
public surface and `Reply` layout — built for long-running batch and pipeline
workloads where rate limiting and cost accounting are what matter. Defaults
target the open-weight `openai/gpt-oss-20b` model.

## Features

- `chat` / `chat_async` against `/openai/v1/chat/completions` with HTTP
  keep-alive pooling.
- `reasoning_effort` passthrough for gpt-oss reasoning models (defaults to
  `"low"` for fast, cheap output). The reasoning channel is **dropped** from
  `Reply.text` — you get the final answer only.
- `response_format` passthrough for **JSON Object** and strict **JSON Schema**
  structured output — guaranteed-parseable JSON.
- Per-client sliding-window RPM semaphore shared across concurrent calls.
- Per-reply token + USD cost accounting (uncached input, cached reads, output)
  against a bundled per-model price table.
- `Budget` wrapper that throws `BudgetExceeded` on cap.
- `retry-after`-aware 429 handling; bounded exponential backoff on 5xx.
- Stub-friendly: body-building and reply-parsing are pure functions, so tests
  run with no network and no API key.
- `Base.show` never prints the API key.

## Install

While pre-1.0, use as a git dependency:

```julia
using Pkg
Pkg.add(url="https://github.com/PelehAI/GroqClient.jl")
```

Set your API key in the environment:

```bash
export GROQ_API_KEY=gsk_...
```

## Quick start

```julia
using GroqClient

c = Client(
    api_key       = ENV["GROQ_API_KEY"],
    model_default = "openai/gpt-oss-20b",
    rpm           = 30,
)

reply = chat(c;
    system     = "You are a helpful assistant.",
    messages   = [(:user, "Say hi.")],
    max_tokens = 64,
)
@show reply.text reply.cost_usd reply.input_tokens reply.output_tokens
```

`messages` accepts `Msg`, `(:user, "...")` tuples, or `:user => "..."` pairs.
`system` accepts `String`, `SystemPrompt(text)`, or `(text="...",)` and is sent
as a leading `role:"system"` message.

## Reasoning models (gpt-oss)

gpt-oss emits a separate reasoning channel. This client reads only the final
answer (`message.content`) into `Reply.text` and ignores `message.reasoning`.
Tune depth vs. speed/cost with `reasoning_effort`:

```julia
reply = chat(c;
    messages         = [(:user, "Plan a 3-step outline.")],
    max_tokens       = 512,
    reasoning_effort = "low",   # "low" (default) | "medium" | "high"
)
```

## Structured output (JSON)

For pipelines that parse the model's output, ask for schema-constrained JSON.
`strict = true` uses constrained decoding — the output is guaranteed valid
against your schema:

```julia
schema = Dict(
    "type" => "json_schema",
    "json_schema" => Dict(
        "name"   => "outline",
        "strict" => true,
        "schema" => Dict(
            "type" => "object",
            "properties" => Dict("steps" => Dict("type" => "array",
                                                 "items" => Dict("type" => "string"))),
            "required" => ["steps"],
            "additionalProperties" => false,
        ),
    ),
)

reply = chat(c;
    messages        = [(:user, "Outline a talk on caching.")],
    max_tokens      = 512,
    response_format = schema,
)
# reply.text is valid JSON matching the schema
```

`Dict("type" => "json_object")` is the looser mode (valid JSON syntax, no
schema enforcement) and works on all models.

## Caching

Groq does **automatic** prompt caching server-side — there is no per-block
`cache_control` marker to set. Cache hits show up as `reply.cached_read_tokens`
and are billed at the discounted cache-read rate; `reply.cached_write_tokens`
is always 0 (no write surcharge). The `cache` flag on `Msg`/`SystemPrompt`
exists only for signature parity with `AnthropicClient.jl` and is ignored.

## Concurrency + RPM throttling

`chat_async` returns a `Task` that runs on a background thread. Many concurrent
tasks share one rate budget — the per-client sliding-window semaphore blocks
tasks that would exceed `rpm` requests in the trailing 60s.

```julia
tasks   = [chat_async(c; messages=[(:user, "Q$i")], max_tokens=32) for i in 1:20]
replies = fetch.(tasks)
```

## Cost accounting + budgets

Each `Reply` carries token counts and a USD cost computed against the bundled
price table. Use `known_models()` to list what's billable; update
`src/pricing.jl` when Groq changes pricing or you add models.

```julia
budget = Budget(c; max_usd = 0.10)
for prompt in prompts
    reply = chat(budget; messages=[(:user, prompt)], max_tokens=128)
    # raises BudgetExceeded once spent_usd(budget) crosses max_usd
end
@show spent_usd(budget)
```

## Stub mode (no API key)

```julia
c = Client(api_key="", rpm=30)
has_key(c)   # false
```

Library code can degrade to identity passes / placeholders without a key.
Calling `chat` on a keyless client throws — guard with `has_key`.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

All tests are pure-function / wiring-only — no live API calls.

## Roadmap

- Streaming (SSE) responses
- Tool use / function calling
- `service_tier` (`flex` / `performance`) selection
- More Groq-hosted models in the price table

## Used by

- [peleh.ai](https://peleh.ai) — academic paper to slide deck.

## License

MIT. See `LICENSE`.
