# Budget polymorphism + bookkeeping for cap-enforced sessions.
#
# Callers that accept either a Client or a Budget (e.g. a pipeline's phase
# entry points that keep `client` untyped) need `has_key` to resolve through
# a Budget wrapper. Forward to the wrapped Client.
has_key(b::Budget) = has_key(b.client)

# `chat(client; ...)`  → no cap, just bills.
# `chat(budget; ...)`  → checks cap before issuing, throws BudgetExceeded if
#                        adding this call's actual cost would push over. We
#                        only know the cost AFTER the call, so the check is
#                        post-hoc: if the call pushes us over, we record the
#                        spend (the call happened, you got the response) but
#                        the next call will refuse. There's also a pre-check
#                        that refuses if we're already at/over cap.

"""
    chat(budget::Budget; kwargs...) -> Reply

Like `chat(client; ...)` but enforces `budget.max_usd`. Throws
`BudgetExceeded` BEFORE the call if we're already over cap; throws AFTER the
call if the call pushed us over. The reply is always recorded in the budget.
"""
function chat(b::Budget; kwargs...)
    lock(b.lock) do
        b.used_usd >= b.max_usd &&
            throw(BudgetExceeded(b.used_usd, b.max_usd, 0.0))
    end
    rep = chat(b.client; kwargs...)
    lock(b.lock) do
        b.used_usd += rep.cost_usd
        if b.used_usd > b.max_usd
            # The call has already happened — caller still gets the reply if
            # they catch — but we signal that future calls should not proceed.
            throw(BudgetExceeded(b.used_usd, b.max_usd, rep.cost_usd))
        end
    end
    return rep
end

"""
    chat_async(budget::Budget; kwargs...) -> Task{Reply}
"""
function chat_async(b::Budget; kwargs...)
    Threads.@spawn chat(b; kwargs...)
end
