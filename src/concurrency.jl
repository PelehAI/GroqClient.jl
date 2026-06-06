# RPM-aware semaphore for parallel calls.
#
# Groq enforces requests-per-minute caps per API key. Many simultaneous tasks
# must share one rate budget. We use a sliding-window approach: we keep a deque
# of timestamps of recent calls; before issuing a new call we evict entries
# older than 60s and block if the deque is at the cap.

const _RPM_WINDOW_SECONDS = 60.0

"""
    await_slot!(client)

Block until the client can make another API call without exceeding its RPM
cap. Updates the sliding window in-place. Safe for concurrent callers thanks
to the per-client lock.
"""
function await_slot!(client::Client)
    while true
        wait_secs = lock(client.rpm_lock) do
            now = time()
            cutoff = now - _RPM_WINDOW_SECONDS
            w = client.rpm_window
            # Evict expired
            i = 1
            while i <= length(w) && w[i] < cutoff
                i += 1
            end
            i > 1 && deleteat!(w, 1:i-1)
            if length(w) < client.rpm
                push!(w, now)
                return 0.0
            else
                # Sleep until the oldest in-window entry ages out.
                return (w[1] + _RPM_WINDOW_SECONDS) - now + 0.01
            end
        end
        wait_secs <= 0 && return nothing
        sleep(wait_secs)
    end
end
