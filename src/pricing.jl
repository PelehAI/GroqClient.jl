# Groq per-model pricing (USD per million tokens). Numbers from Groq's own
# model pages (console.groq.com/docs/models) as of 2026-06. Update as Groq
# changes pricing or you add models.
#
# Groq does automatic prompt caching with a discounted `cache_read` rate and
# NO cache-write surcharge, so `cache_write` is 0 for every model.

const _PRICES = Dict{String, NamedTuple{(:input, :cache_read, :cache_write, :output), NTuple{4, Float64}}}(
    # OpenAI open-weight models, hosted on GroqCloud
    "openai/gpt-oss-20b"  => (input=0.075, cache_read=0.037, cache_write=0.0, output=0.30),
)

"""
    price_for(model) -> NamedTuple

Returns the per-million-token rates for `model`. Falls back to `gpt-oss-20b`
rates (with a `@warn`) if the exact model isn't in the table — update
`_PRICES` when you add a model.
"""
function price_for(model::AbstractString)
    if haskey(_PRICES, model)
        return _PRICES[model]
    end
    @warn "GroqClient: unknown model in pricing table; using gpt-oss-20b rates as fallback" model
    return _PRICES["openai/gpt-oss-20b"]
end

"""
    calc_cost(model, input_tokens, cached_read, cached_write, output_tokens) -> Float64

USD cost for a single Groq call. All token counts are integers. `input_tokens`
is the *uncached* prompt remainder; `cached_read` is the cache-hit portion.
"""
function calc_cost(model::AbstractString, input::Integer, cached_read::Integer,
                   cached_write::Integer, output::Integer)
    p = price_for(model)
    return (input        * p.input        +
            cached_read  * p.cache_read   +
            cached_write * p.cache_write  +
            output       * p.output) / 1_000_000
end

"""
    known_models() -> Vector{String}

Sorted list of every model identifier in the bundled pricing table.
"""
known_models() = sort!(collect(keys(_PRICES)))
