using Test
using GroqClient
using GroqClient: build_body, parse_reply, calc_cost, _serialize_messages,
                  Msg, SystemPrompt, to_msg, to_system,
                  await_slot!, _retry_after_seconds, price_for
using HTTP
using JSON3

# All tests in this file are pure functions / wiring-only. None hit the
# real Groq API.

@testset "GroqClient pure-function unit tests" begin

    @testset "Msg normalization" begin
        m1 = to_msg(Msg(:user, "hi"))
        m2 = to_msg((:user, "hi"))
        m3 = to_msg(:user => "hi")
        @test m1.role == :user && m1.content == "hi"
        @test m2.role == :user && m2.content == "hi"
        @test m3.role == :user && m3.content == "hi"
        @test_throws ArgumentError Msg(:system, "nope")  # system goes via SystemPrompt
    end

    @testset "SystemPrompt normalization" begin
        @test to_system(nothing) === nothing
        sp = to_system("hello")
        @test sp.text == "hello" && sp.cache == false
        sp2 = to_system((text="hi", cache=true))
        @test sp2.text == "hi" && sp2.cache == true   # cache flag stored but no-op
        sp3 = to_system(SystemPrompt("X"))
        @test sp3.text == "X"
    end

    @testset "Pricing table + calc_cost" begin
        p = price_for("openai/gpt-oss-20b")
        @test p.input == 0.075
        @test p.output == 0.30
        @test p.cache_write == 0.0
        # 1M uncached input + 1M output = 0.075 + 0.30 = 0.375
        c = calc_cost("openai/gpt-oss-20b", 1_000_000, 0, 0, 1_000_000)
        @test c ≈ 0.375 atol=1e-9
        # cached-read pricing: 1M cached input @ 0.037
        c2 = calc_cost("openai/gpt-oss-20b", 0, 1_000_000, 0, 0)
        @test c2 ≈ 0.037 atol=1e-9
        # Unknown model warns and falls back to gpt-oss-20b rates
        c3 = (@test_logs (:warn, r"unknown model") calc_cost("groq/whatever", 1000, 0, 0, 1000))
        @test c3 ≈ (1000*0.075 + 1000*0.30)/1_000_000 atol=1e-12
    end

    @testset "build_body — simple call" begin
        client = Client(api_key="dummy", model_default="openai/gpt-oss-20b", rpm=30)
        body = build_body(client;
            system = nothing,
            messages = [Msg(:user, "hi")],
            max_tokens = 64,
            model = nothing,
            temperature = nothing,
            reasoning_effort = nothing,
            response_format = nothing,
        )
        @test body["model"] == "openai/gpt-oss-20b"
        @test body["max_completion_tokens"] == 64
        @test !haskey(body, "max_tokens")            # Groq uses max_completion_tokens
        @test length(body["messages"]) == 1
        @test body["messages"][1]["role"] == "user"
        @test body["messages"][1]["content"] == "hi"
        @test !haskey(body, "temperature")
        @test !haskey(body, "reasoning_effort")
        @test !haskey(body, "response_format")
    end

    @testset "build_body — system + temperature + reasoning + json schema" begin
        client = Client(api_key="dummy", model_default="openai/gpt-oss-20b", rpm=30)
        schema = Dict("type" => "json_schema",
                      "json_schema" => Dict("name" => "out", "strict" => true,
                                            "schema" => Dict("type" => "object")))
        body = build_body(client;
            system = SystemPrompt("You are X."),
            messages = [Msg(:user, "hi"), Msg(:assistant, "hello"), Msg(:user, "again")],
            max_tokens = 100,
            model = "openai/gpt-oss-120b",
            temperature = 0.3,
            reasoning_effort = "low",
            response_format = schema,
        )
        @test body["model"] == "openai/gpt-oss-120b"
        @test body["temperature"] == 0.3
        @test body["reasoning_effort"] == "low"
        @test body["response_format"]["type"] == "json_schema"
        # system is prepended as a role:"system" message → 4 total
        @test length(body["messages"]) == 4
        @test body["messages"][1]["role"] == "system"
        @test body["messages"][1]["content"] == "You are X."
        @test body["messages"][2]["role"] == "user"
        @test body["messages"][3]["role"] == "assistant"
    end

    @testset "parse_reply — content extracted, reasoning ignored, usage mapped" begin
        sample = JSON3.read("""
        {"model":"openai/gpt-oss-20b",
         "choices":[{"message":{"role":"assistant",
                                "content":"{\\"ok\\":true}",
                                "reasoning":"let me think... internal CoT"},
                     "finish_reason":"stop"}],
         "usage":{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050,
                  "prompt_tokens_details":{"cached_tokens":200}}}
        """)
        r = parse_reply(sample, "openai/gpt-oss-20b")
        @test r.text == "{\"ok\":true}"          # final answer only
        @test !occursin("CoT", r.text)            # reasoning channel dropped
        @test r.stop_reason == :end_turn
        @test r.input_tokens == 800               # 1000 − 200 cached
        @test r.cached_read_tokens == 200
        @test r.cached_write_tokens == 0
        @test r.output_tokens == 50
        @test r.cost_usd ≈ (800*0.075 + 200*0.037 + 50*0.30)/1_000_000 atol=1e-12
    end

    @testset "parse_reply — finish_reason mapping" begin
        mk(fr) = JSON3.read("""{"model":"openai/gpt-oss-20b","choices":[{"message":{"content":"x"},"finish_reason":"$fr"}],"usage":{}}""")
        @test parse_reply(mk("stop"),       "m").stop_reason == :end_turn
        @test parse_reply(mk("length"),     "m").stop_reason == :max_tokens
        @test parse_reply(mk("tool_calls"), "m").stop_reason == :tool_use
        @test parse_reply(mk("content_filter"), "m").stop_reason == :other
    end

    @testset "RPM semaphore admits within cap" begin
        c = Client(api_key="dummy", rpm=3)
        t = @elapsed (await_slot!(c); await_slot!(c); await_slot!(c))
        @test t < 0.5   # three calls, cap of 3 → no blocking
    end

    @testset "has_key / stub mode" begin
        @test has_key(Client(api_key="x")) == true
        @test has_key(Client(api_key="")) == false
    end
end

@testset "GroqClient — health probe (offline)" begin
    cls = GroqClient._health_status_from
    @test cls(429, "") == :quota
    @test cls(400, "RESOURCE_EXHAUSTED: quota exceeded") == :quota
    @test cls(401, "unauthorized") == :auth
    @test cls(403, "permission_denied") == :auth
    @test cls(400, "API_KEY_INVALID: invalid api key") == :auth
    @test cls(400, "credit balance is too low") == :billing
    @test cls(402, "") == :billing
    @test cls(400, "bad model id") == :bad_request
    @test cls(503, "") == :server
    @test cls(0, "connection refused") == :network
    @test cls(418, "teapot") == :error

    @test GroqClient._median_int(Int[]) == 0
    @test GroqClient._median_int([5]) == 5
    @test GroqClient._median_int([3, 1, 2]) == 2
    @test GroqClient._median_int([2, 2, 4, 4]) == 3

    h = healthcheck(Client(api_key=""))
    @test h.ok == false
    @test h.status == :no_key
    @test h.http_status == 0
    s = speedtest(Client(api_key=""); n=3)
    @test s.ok == 0 && s.failed == 3 && s.throughput_rps == 0.0

    io = IOBuffer(); show(io, HealthStatus(true, :ok, 200, 42, "m", "ok"))
    @test occursin("OK", String(take!(io)))
    io = IOBuffer(); show(io, SpeedResult(5, 5, 0, 0, 1000, 5.0, 10, 20, 30, "m"))
    @test occursin("req/s", String(take!(io)))
end
