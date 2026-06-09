# Live key/endpoint health probe.
#
# `has_key` only checks that an api_key string is set — it cannot tell a good
# key from a revoked one, or a funded account from an exhausted one. `healthcheck`
# issues ONE minimal real call (4 output tokens, no retries) and classifies the
# outcome so an operator dashboard can show green/red and say *why*.

"""
    HealthStatus

Outcome of a [`healthcheck`](@ref). `ok` is true only when the provider answered
a real request. `status` is a coarse, machine-friendly classification:

- `:ok`          — the API answered (key valid, account funded).
- `:no_key`      — no api_key configured on the client.
- `:auth`        — key rejected / unauthorized / permission denied.
- `:quota`       — rate-limited or quota/credits exhausted (429 / RESOURCE_EXHAUSTED).
- `:billing`     — billing problem (low credit balance, suspended, payment).
- `:bad_request` — 400 that isn't a key/billing problem (e.g. bad model id).
- `:server`      — provider 5xx.
- `:network`     — couldn't reach the endpoint at all.
- `:error`       — anything else.

`http_status` is the last HTTP status (0 if the call never reached the server).
`detail` is a short, secret-free human string. Never contains the api_key.
"""
struct HealthStatus
    ok::Bool
    status::Symbol
    http_status::Int
    latency_ms::Int
    model::String
    detail::String
end

function Base.show(io::IO, h::HealthStatus)
    print(io, "HealthStatus(", h.ok ? "OK" : "DOWN", " :", h.status,
              h.http_status == 0 ? "" : string(" http=", h.http_status),
              ", ", h.latency_ms, "ms, model=", repr(h.model), ")")
end

# Pure classifier: map (HTTP status, response body) → a HealthStatus symbol.
# Body is inspected first because the same status code carries different meanings
# (e.g. Anthropic returns 400 with "credit balance is too low" for billing).
# Tested as a unit.
function _health_status_from(status::Integer, body::AbstractString)
    b = lowercase(body)
    if occursin("api_key_invalid", b) || occursin("api key not valid", b) ||
       occursin("invalid api key", b) || occursin("invalid x-api-key", b) ||
       occursin("permission_denied", b) || occursin("unauthorized", b) ||
       occursin("authentication", b)
        return :auth
    elseif occursin("credit balance", b) || occursin("billing", b) ||
           occursin("suspended", b) || occursin("payment", b)
        return :billing
    elseif status == 429 || occursin("resource_exhausted", b) ||
           occursin("quota", b) || occursin("rate limit", b)
        return :quota
    elseif status in (401, 403)
        return :auth
    elseif status == 402
        return :billing
    elseif status == 400
        return :bad_request
    elseif 500 <= status < 600
        return :server
    elseif status == 0
        return :network
    else
        return :error
    end
end

_health_detail(s::AbstractString; n::Integer = 180) =
    (s = strip(s); length(s) <= n ? String(s) : string(first(s, n), "…"))

"""
    healthcheck(client; model=nothing, max_tokens=4) -> HealthStatus

Probe the provider with ONE minimal real call and classify the result. Costs a
few output tokens on success; fails fast on error (no retries). Never throws —
every failure mode is returned as a [`HealthStatus`](@ref). Safe to call from an
operator dashboard.
"""
function healthcheck(client::Client; model::Union{Nothing,AbstractString}=nothing,
                     max_tokens::Integer=4)
    mdl = model === nothing ? client.model_default : String(model)
    has_key(client) || return HealthStatus(false, :no_key, 0, 0, mdl, "no API key configured")
    t0 = time()
    try
        r = chat(client; messages=[(:user, "ping")], max_tokens=max_tokens,
                 model=mdl, max_retries=0)
        return HealthStatus(true, :ok, 200, round(Int, (time()-t0)*1000), r.model, "ok")
    catch e
        ms = round(Int, (time()-t0)*1000)
        if e isa GroqAPIError
            return HealthStatus(false, _health_status_from(e.status, e.response_body),
                                e.status, ms, mdl,
                                _health_detail(isempty(e.response_body) ? e.message : e.response_body))
        elseif e isa ArgumentError
            return HealthStatus(false, :no_key, 0, ms, mdl, "no API key configured")
        else
            return HealthStatus(false, :network, 0, ms, mdl, _health_detail(sprint(showerror, e)))
        end
    end
end

"""
    SpeedResult

Outcome of a [`speedtest`](@ref). `throughput_rps` is successful calls per wall
second under the client's configured `rpm` cap — so a low number with a nonzero
`rate_limited` count means you're hitting the provider's limit, not that the
network is slow. Latencies are per-successful-call, in milliseconds.
"""
struct SpeedResult
    n::Int
    ok::Int
    rate_limited::Int
    failed::Int
    wall_ms::Int
    throughput_rps::Float64
    latency_min_ms::Int
    latency_median_ms::Int
    latency_max_ms::Int
    model::String
end

function Base.show(io::IO, s::SpeedResult)
    print(io, "SpeedResult(", s.ok, "/", s.n, " ok",
              s.rate_limited == 0 ? "" : string(", ", s.rate_limited, " rate-limited"),
              s.failed == 0 ? "" : string(", ", s.failed, " failed"),
              ", ", round(s.throughput_rps; digits=2), " req/s",
              ", lat ", s.latency_min_ms, "/", s.latency_median_ms, "/", s.latency_max_ms,
              "ms min/med/max)")
end

_median_int(v::AbstractVector{<:Integer}) =
    isempty(v) ? 0 : (s = sort(v); n = length(s);
        isodd(n) ? s[(n+1)÷2] : round(Int, (s[n÷2] + s[n÷2+1]) / 2))

"""
    speedtest(client; n=5, model=nothing, max_tokens=4) -> SpeedResult

Fire `n` minimal calls concurrently and measure latency + achieved throughput.
Calls share the client's `rpm` cap (so this also reveals how the limit throttles
you). Never throws; 429s are counted in `rate_limited`, other errors in `failed`.
Costs a few output tokens per successful call — keep `n` small.
"""
function speedtest(client::Client; n::Integer=5,
                   model::Union{Nothing,AbstractString}=nothing,
                   max_tokens::Integer=4)
    mdl = model === nothing ? client.model_default : String(model)
    has_key(client) && n > 0 || return SpeedResult(Int(n), 0, 0, Int(n), 0, 0.0, 0, 0, 0, mdl)
    t_start = time()
    tasks = map(1:n) do _
        Threads.@spawn begin
            t0 = time()
            try
                chat(client; messages=[(:user, "ping")], max_tokens=max_tokens,
                     model=mdl, max_retries=0)
                (:ok, round(Int, (time()-t0)*1000))
            catch e
                code = e isa GroqAPIError ? e.status : 0
                (code == 429 ? :rl : :fail, round(Int, (time()-t0)*1000))
            end
        end
    end
    lat = Int[]; ok = 0; rl = 0; fail = 0
    for t in tasks
        kind, ms = fetch(t)
        kind === :ok ? (ok += 1; push!(lat, ms)) : kind === :rl ? (rl += 1) : (fail += 1)
    end
    wall = round(Int, (time()-t_start)*1000)
    rps  = wall > 0 ? ok / (wall/1000) : 0.0
    SpeedResult(Int(n), ok, rl, fail, wall, rps,
                isempty(lat) ? 0 : minimum(lat), _median_int(lat),
                isempty(lat) ? 0 : maximum(lat), mdl)
end
