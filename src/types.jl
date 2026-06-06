# Core types: Client, Msg, SystemPrompt, Reply, Budget, BudgetExceeded.
# Mirrors AnthropicClient.jl so the two clients are interchangeable behind a
# common interface.

"""
    Client(; api_key, model_default, rpm, base_url, timeout)

A reusable client for Groq's OpenAI-compatible Chat Completions API. Maintains
an HTTP keep-alive connection pool and a rate-limit semaphore shared across all
calls.

Fields:
- `api_key::String`           — from `GROQ_API_KEY` env by default.
- `model_default::String`     — used when a call doesn't override. Defaults to
                                `openai/gpt-oss-20b`.
- `rpm::Int`                  — requests-per-minute cap. Groq's free tier is
                                generous; raise this once you know your tier.
- `base_url::String`          — typically `https://api.groq.com/openai/v1`.
- `timeout::Int`              — seconds, per-call HTTP timeout.
- internal: rate-limit window state + lock.
"""
struct Client
    api_key::String
    model_default::String
    rpm::Int
    base_url::String
    timeout::Int
    # RPM sliding window: timestamps of recent calls. Vector is mutable
    # through its reference, so no Ref wrapper needed.
    rpm_window::Vector{Float64}
    rpm_lock::ReentrantLock
end

function Client(;
    api_key::AbstractString       = get(ENV, "GROQ_API_KEY", ""),
    model_default::AbstractString = "openai/gpt-oss-20b",
    rpm::Integer                  = 30,
    base_url::AbstractString      = "https://api.groq.com/openai/v1",
    timeout::Integer              = 120,
)
    return Client(
        String(api_key),
        String(model_default),
        Int(rpm),
        String(base_url),
        Int(timeout),
        Float64[],
        ReentrantLock(),
    )
end

"Has-key sanity check. Stub mode is when no api_key is set."
has_key(c::Client) = !isempty(c.api_key)

# Custom show — never leak the api_key in a repr or error message.
function Base.show(io::IO, c::Client)
    masked = isempty(c.api_key) ? "<unset>" :
             length(c.api_key) <= 8 ? "***" :
             string(first(c.api_key, 4), "…", last(c.api_key, 4))
    print(io, "GroqClient.Client(api_key=", masked,
              ", model_default=", repr(c.model_default),
              ", rpm=", c.rpm, ")")
end

# A message is one (role, content) pair. The `cache` flag exists only for
# signature parity with AnthropicClient — Groq does automatic prompt caching
# server-side and has no per-block `cache_control` marker, so it is a no-op.
"""
    Msg(role, content; cache=false)

One message in a conversation. `role` is `:user` or `:assistant`. `content` is
a String. `cache` is accepted for interface parity but ignored (Groq caches
automatically).
"""
struct Msg
    role::Symbol     # :user or :assistant
    content::String
    cache::Bool
end
function Msg(role::Symbol, content::AbstractString; cache::Bool=false)
    role in (:user, :assistant) ||
        throw(ArgumentError("Msg: role must be :user or :assistant, got :$role"))
    return Msg(role, String(content), cache)
end

# Sugar so callers can use tuples or Pair where clarity wins.
to_msg(m::Msg) = m
to_msg(p::Tuple{Symbol,<:AbstractString}) = Msg(p[1], p[2])
to_msg(p::Pair{Symbol,<:AbstractString})  = Msg(p[1], p[2])

# System prompt can be plain String, or NamedTuple (text=...,). Serialized as a
# leading role:"system" message. `cache` is accepted but ignored (see above).
"""
    SystemPrompt(text; cache=false)

System prompt block, emitted as a leading `role:"system"` message. `cache` is
accepted for parity with AnthropicClient but ignored (Groq caches
automatically — there is no explicit cache breakpoint).
"""
struct SystemPrompt
    text::String
    cache::Bool
end
SystemPrompt(text::AbstractString; cache::Bool=false) = SystemPrompt(String(text), cache)

to_system(::Nothing) = nothing
to_system(sp::SystemPrompt) = sp
to_system(s::AbstractString) = SystemPrompt(String(s), false)
to_system(nt::NamedTuple) = SystemPrompt(String(nt.text), get(nt, :cache, false))

"""
    Reply

Result of one chat call. Numbers come from Groq's `usage` field; `cost_usd` is
computed via `calc_cost`. Field layout matches `AnthropicClient.Reply` so the
two are interchangeable downstream.

Note: `cached_write_tokens` is always 0 (Groq has no cache-write surcharge),
and `cached_read_tokens` carries `prompt_tokens_details.cached_tokens` when
present. `input_tokens` is the *uncached* prompt remainder.
"""
struct Reply
    text::String
    model::String
    stop_reason::Symbol             # :end_turn | :max_tokens | :tool_use | :other
    input_tokens::Int               # uncached input (prompt_tokens − cached)
    cached_read_tokens::Int         # cache-hit input (cheap)
    cached_write_tokens::Int        # always 0 on Groq
    output_tokens::Int
    cost_usd::Float64
    raw::Any                        # full JSON response (JSON3.Object), kept for debugging
end

function Base.show(io::IO, r::Reply)
    snippet = length(r.text) <= 40 ? r.text : string(first(r.text, 37), "...")
    print(io, "Reply(", repr(snippet),
              ", model=", repr(r.model),
              ", in=", r.input_tokens,
              ", out=", r.output_tokens,
              ", \$", round(r.cost_usd; digits=6), ")")
end

"""
    Budget(client; max_usd)

Wrap a client with a per-session spend cap. Calls via `chat(budget; ...)`
deduct from the budget; over-cap throws `BudgetExceeded`.
"""
mutable struct Budget
    client::Client
    max_usd::Float64
    used_usd::Float64
    lock::ReentrantLock
end
Budget(client::Client; max_usd::Real = 1.0) =
    Budget(client, Float64(max_usd), 0.0, ReentrantLock())

spent_usd(b::Budget) = b.used_usd

struct BudgetExceeded <: Exception
    used_usd::Float64
    max_usd::Float64
    attempt_cost_usd::Float64
end
Base.showerror(io::IO, e::BudgetExceeded) = print(io,
    "BudgetExceeded: already spent \$", round(e.used_usd; digits=4),
    ", cap is \$", round(e.max_usd; digits=4),
    ", this call would add \$", round(e.attempt_cost_usd; digits=4))

"""
    GroqAPIError(status, message, response_body)

Thrown by `chat` when the Groq API returns a non-recoverable HTTP error, or
when retries are exhausted on 429 / 5xx. `status` is the last seen HTTP status
(0 if the failure was network-level), `message` is a short human-readable
label, and `response_body` carries the raw response body if any.
"""
struct GroqAPIError <: Exception
    status::Int
    message::String
    response_body::String
end
GroqAPIError(status::Integer, message::AbstractString) =
    GroqAPIError(Int(status), String(message), "")
Base.showerror(io::IO, e::GroqAPIError) = print(io,
    "GroqAPIError(status=", e.status, "): ", e.message,
    isempty(e.response_body) ? "" :
        string(" — body: ", first(e.response_body, 200),
               length(e.response_body) > 200 ? "..." : ""))
