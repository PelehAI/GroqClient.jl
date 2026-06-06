# HTTP wiring for Groq's /openai/v1/chat/completions endpoint (OpenAI-compatible).
#
# All API access goes through `post_chat` which:
# - waits for an RPM slot
# - issues an HTTP POST with a Bearer auth header
# - honours `retry-after` on 429
# - retries up to `max_retries` on 429/5xx

# Pure function: build the request body. Tested as a unit.
"""
    build_body(client; system, messages, max_tokens, model, temperature,
               reasoning_effort, response_format)

Build the JSON body for `/chat/completions`. Returns a `Dict{String,Any}` ready
to hand to `JSON3.write`. Pure function; no I/O. A non-`nothing` `system` is
emitted as a leading `role:"system"` message.
"""
function build_body(client::Client;
    system::Union{Nothing, SystemPrompt},
    messages::AbstractVector{Msg},
    max_tokens::Integer,
    model::Union{Nothing, AbstractString},
    temperature::Union{Nothing, Real},
    reasoning_effort::Union{Nothing, AbstractString},
    response_format::Union{Nothing, AbstractDict},
)
    body = Dict{String,Any}(
        "model"                 => model === nothing ? client.model_default : String(model),
        "messages"              => _serialize_messages(system, messages),
        "max_completion_tokens" => Int(max_tokens),
    )
    temperature      !== nothing && (body["temperature"]      = Float64(temperature))
    reasoning_effort !== nothing && (body["reasoning_effort"] = String(reasoning_effort))
    response_format  !== nothing && (body["response_format"]  = response_format)
    return body
end

function _serialize_messages(system::Union{Nothing, SystemPrompt},
                             messages::AbstractVector{Msg})
    out = Vector{Dict{String,Any}}()
    if system !== nothing
        push!(out, Dict{String,Any}("role" => "system", "content" => system.text))
    end
    for m in messages
        # Role is validated at Msg construction time, so this is total.
        role = m.role === :user ? "user" : "assistant"
        push!(out, Dict{String,Any}("role" => role, "content" => m.content))
    end
    return out
end

# Pure function: turn a Groq chat-completions response (already parsed
# JSON3.Object) into a Reply. Tested as a unit.
"""
    parse_reply(json, model_requested)

Parse Groq's response object into a `Reply`. Reads only the final answer
(`choices[1].message.content`); the reasoning channel
(`choices[1].message.reasoning`) is deliberately ignored. `model_requested`
is used for cost if the response doesn't echo the model name.
"""
function parse_reply(json, model_requested::AbstractString)
    model_in_resp = haskey(json, "model") ? String(json["model"]) : String(model_requested)

    text = ""
    stop_raw = "stop"
    choices = get(json, "choices", ())
    if !isempty(choices)
        ch  = choices[1]
        msg = get(ch, "message", Dict{String,Any}())
        c   = get(msg, "content", "")
        text = c === nothing ? "" : String(c)
        stop_raw = String(get(ch, "finish_reason", "stop"))
    end

    stop_reason = stop_raw == "stop"       ? :end_turn :
                  stop_raw == "length"     ? :max_tokens :
                  stop_raw == "tool_calls" ? :tool_use :
                                             :other

    usage             = get(json, "usage", Dict{String,Any}())
    prompt_tokens     = Int(get(usage, "prompt_tokens", 0))
    completion_tokens = Int(get(usage, "completion_tokens", 0))
    # `cached_tokens` (when present) is a SUBSET of prompt_tokens, so the
    # full-price uncached input is prompt_tokens − cached.
    details           = get(usage, "prompt_tokens_details", Dict{String,Any}())
    cached_read       = Int(get(details, "cached_tokens", 0))
    input_uncached    = max(prompt_tokens - cached_read, 0)

    cost_usd = calc_cost(model_in_resp, input_uncached, cached_read, 0, completion_tokens)

    return Reply(text, model_in_resp, stop_reason,
                 input_uncached, cached_read, 0,
                 completion_tokens, cost_usd, json)
end

# ---- HTTP transport --------------------------------------------------------

"""
    post_chat(client, body; max_retries=3)

Issue one POST to `/chat/completions` with the Bearer auth header. Handles
`retry-after` on 429, retries on 5xx. Returns parsed JSON3.Object.
"""
function post_chat(client::Client, body::AbstractDict; max_retries::Integer=3)
    has_key(client) || throw(ArgumentError("GROQ_API_KEY not set on client"))
    url = string(client.base_url, "/chat/completions")
    headers = [
        "Authorization" => string("Bearer ", client.api_key),
        "Content-Type"  => "application/json",
    ]
    body_str = JSON3.write(body)

    attempt = 0
    while true
        attempt += 1
        await_slot!(client)
        local resp
        try
            resp = HTTP.post(url, headers, body_str;
                             readtimeout=client.timeout, retry=false, status_exception=false)
        catch e
            if attempt > max_retries
                rethrow(e)
            end
            # Network-level errors → backoff + retry.
            sleep(min(2.0 ^ attempt, 30.0))
            continue
        end
        if resp.status == 200
            return JSON3.read(resp.body)
        elseif resp.status == 429
            retry_after = _retry_after_seconds(resp)
            if attempt > max_retries
                throw(GroqAPIError(429,
                    "rate limit exceeded after $max_retries retries",
                    String(resp.body)))
            end
            sleep(retry_after)
            continue
        elseif 500 <= resp.status < 600
            if attempt > max_retries
                throw(GroqAPIError(resp.status,
                    "server error after $max_retries retries",
                    String(resp.body)))
            end
            sleep(min(2.0 ^ attempt, 30.0))
            continue
        else
            # 4xx that isn't 429 — surface the body, don't retry.
            throw(GroqAPIError(resp.status,
                "non-recoverable HTTP error from /chat/completions",
                String(resp.body)))
        end
    end
end

function _retry_after_seconds(resp::HTTP.Response)
    for (k, v) in resp.headers
        lowercase(k) == "retry-after" || continue
        secs = tryparse(Float64, v)
        secs !== nothing && secs > 0 && return secs
    end
    return 5.0  # conservative default
end
