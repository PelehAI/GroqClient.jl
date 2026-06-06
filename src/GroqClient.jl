"""
    GroqClient

Julia client for [Groq's](https://groq.com) OpenAI-compatible Chat Completions
API. Built for the same long-running batch / pipeline workloads as its sibling
`AnthropicClient.jl`: rate limiting, cost accounting, and a stub-friendly pure
core. Defaults target the open-weight `openai/gpt-oss-20b` model.

Construct a [`Client`](@ref), call [`chat`](@ref) or [`chat_async`](@ref),
inspect [`Reply`](@ref). See the README for the full guide.
"""
module GroqClient

# Minimal, fast Julia client for Groq's /openai/v1/chat/completions endpoint.
# Mirrors the structure of AnthropicClient.jl so the two are drop-in siblings
# behind a common `chat(client; system, messages, max_tokens) -> Reply` shape.

using HTTP
using JSON3

include("pricing.jl")
include("types.jl")
include("concurrency.jl")
include("groq.jl")

# ------------------- Public API ---------------------------------------------

export Client, has_key,
       Msg, SystemPrompt,
       Reply,
       Budget, BudgetExceeded, spent_usd,
       GroqAPIError,
       chat, chat_async,
       calc_cost, price_for, known_models

"""
    chat(client; system=nothing, messages, max_tokens, model=nothing,
                 temperature=nothing, reasoning_effort="low",
                 response_format=nothing, max_retries=3) -> Reply

Make one Groq Chat Completions call. Returns a `Reply` with text, token
counts, and computed `cost_usd`.

# Arguments
- `system` — `nothing`, a `String`, a `SystemPrompt(text)`, or a `NamedTuple`
  `(text="...",)`. Serialized as a leading `role:"system"` message.
- `messages` — vector of `Msg`, `(:user, "...")` tuples, or `:user => "..."`
  pairs. Roles must be `:user` or `:assistant`.
- `max_tokens` — required. Sent as Groq's `max_completion_tokens`.
- `model` — defaults to `client.model_default` (`openai/gpt-oss-20b`).
- `temperature` — optional Float.
- `reasoning_effort` — gpt-oss reasoning depth: `"low"` (default, fast/cheap),
  `"medium"`, `"high"`. Pass `nothing` to omit the field.
- `response_format` — optional `Dict`, e.g.
  `Dict("type" => "json_object")` or a `json_schema` block for strict,
  schema-constrained JSON output. See the README.
- `max_retries` — for 429 / 5xx retries; default 3.

The model's reasoning channel (`message.reasoning`) is intentionally dropped —
`Reply.text` is only the final answer (`message.content`).
"""
function chat(client::Client;
    system   = nothing,
    messages = nothing,
    max_tokens::Integer,
    model::Union{Nothing, AbstractString}      = nothing,
    temperature::Union{Nothing, Real}          = nothing,
    reasoning_effort::Union{Nothing, AbstractString} = "low",
    response_format::Union{Nothing, AbstractDict}    = nothing,
    max_retries::Integer = 3,
)
    messages === nothing && throw(ArgumentError("chat: `messages` is required"))
    msgs_normalized = Msg[to_msg(m) for m in messages]
    sys_normalized  = to_system(system)
    body = build_body(client;
        system = sys_normalized,
        messages = msgs_normalized,
        max_tokens = max_tokens,
        model = model,
        temperature = temperature,
        reasoning_effort = reasoning_effort,
        response_format = response_format,
    )
    json = post_chat(client, body; max_retries=max_retries)
    return parse_reply(json, model === nothing ? client.model_default : String(model))
end

"""
    chat_async(client; kwargs...) -> Task

Spawn `chat` on a background thread; `fetch(task)` returns the `Reply`.
The task respects the client's RPM cap through the shared semaphore —
many concurrent tasks share one rate budget.
"""
function chat_async(client::Client; kwargs...)
    Threads.@spawn chat(client; kwargs...)
end

# Budget wrapper methods are defined in budget.jl, which is included AFTER
# `chat(client; ...)` is in scope above.
include("budget.jl")

end # module
