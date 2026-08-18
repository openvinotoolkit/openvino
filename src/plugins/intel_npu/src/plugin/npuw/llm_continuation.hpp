// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

#include "logging.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace npuw {

/**
 * @brief Transaction coordinator for continuous prefill.
 *
 * Owns the propose/grant negotiation and the request-level transaction state
 * machine. The caller proposes a common token prefix through the
 * npuw_stored_tokens_state variable state, the coordinator clamps it by every
 * limit the plugin owns and grants an effective keep K. A granted keep is a
 * command, so the next inference must be the matching delta-only prefill.
 *
 * Three states. While IDLE the state reports the committed live token count
 * and accepts one proposal, a reset, or ordinary inference. PENDING means a
 * command is armed or executing: a positive keep demands the delta-only
 * prefill, a zero keep demands a full prefill from position zero. Any failure
 * after mutation started poisons the request as POISONED, where only reset()
 * is accepted: the caller sent only the delta, so recovering by falling back
 * to an ordinary prefill would treat the delta as the whole conversation.
 *
 * The keep rule is K = C * floor(min(K_common, K_max, W) / C), where C is the
 * prefill chunk size, K_max is how many past tokens the chunked prefill can
 * represent, and W is the prefill-produced watermark. The rule is evaluated
 * here at propose time and nowhere else.
 *
 * Note for callers of the variable state: a write is not idempotent. You write
 * X and read back Y <= X, and must always slice at the granted value.
 */
class ContinuationCoordinator {
public:
    enum class Stage : uint8_t {
        IDLE,
        PENDING,
        POISONED,
    };

    ContinuationCoordinator() = default;

    void enable(uint32_t chunk_size, uint32_t max_prompt_len) {
        OPENVINO_ASSERT(chunk_size > 0u && max_prompt_len > chunk_size,
                        "Continuous prefill requires 0 < chunk_size < max_prompt_len, got chunk_size=",
                        chunk_size,
                        " max_prompt_len=",
                        max_prompt_len);
        m_enabled = true;
        m_chunk_size = chunk_size;
        m_k_max = max_prompt_len - chunk_size;
    }

    bool enabled() const {
        return m_enabled;
    }

    Stage stage() const {
        return m_stage;
    }

    // Variable-state side, driven by StoredTokensState.

    // A proposal from set_state(N). Validates, applies the keep rule, stores the
    // result as the pending command, and returns the granted keep. A zero grant
    // arms a reset: preserving zero tokens is a full reset, not a continuation
    // with an empty prefix.
    uint32_t propose(int64_t k_common) {
        OPENVINO_ASSERT(m_enabled, "Continuous prefill is not enabled for this request.");
        OPENVINO_ASSERT(m_stage == Stage::IDLE,
                        "npuw_stored_tokens_state: a proposal is accepted only while the request is idle; "
                        "current state does not accept a new proposal. Complete or reset() the pending "
                        "operation first.");
        OPENVINO_ASSERT(k_common >= 0, "npuw_stored_tokens_state: proposed prefix must be non-negative.");
        // Growth is an error. Validating against total KV capacity instead would let
        // a caller present uninitialised memory as a valid prefix.
        OPENVINO_ASSERT(static_cast<uint64_t>(k_common) <= m_live_tokens,
                        "npuw_stored_tokens_state: proposed prefix (",
                        k_common,
                        ") exceeds the live token count (",
                        m_live_tokens,
                        "). Growth is an error.");

        const uint32_t clamped = std::min({static_cast<uint32_t>(k_common), m_k_max, m_watermark});
        const uint32_t granted = m_chunk_size * (clamped / m_chunk_size);

        m_pending_keep = granted;
        m_stage = Stage::PENDING;
        LOG_DEBUG("Continuous prefill: proposed " << k_common << ", granted " << granted << " (K_max=" << m_k_max
                                                  << ", W=" << m_watermark << ", C=" << m_chunk_size << ")");
        return granted;
    }

    // The grant returned by get_state(), meaning the prefix the plugin will honour.
    int64_t query() const {
        switch (m_stage) {
        case Stage::IDLE:
            return static_cast<int64_t>(m_live_tokens);
        case Stage::PENDING:
            return static_cast<int64_t>(m_pending_keep);
        case Stage::POISONED:
            return 0;
        }
        return 0;
    }

    // reset() is a control operation rather than a proposal. It discards any pending
    // keep and arms a zero keep, idempotently. The physical reset happens on the
    // next validated full prefill. This is also the only recovery from POISONED.
    void request_reset() {
        m_pending_keep = 0u;
        m_stage = Stage::PENDING;
    }

    // Infer-request side.

    // What the next inference must be: no value means ordinary inference, zero
    // means the pending reset's full prefill, positive means the delta-only
    // prefill at exactly that keep. Throws if the request is poisoned.
    std::optional<uint32_t> pending() const {
        if (!m_enabled) {
            return std::nullopt;
        }
        OPENVINO_ASSERT(m_stage != Stage::POISONED,
                        "Continuous prefill: the request is poisoned by a previous failure. "
                        "Call reset() on npuw_stored_tokens_state and re-send the full history.");
        return m_stage == Stage::PENDING ? std::optional<uint32_t>(m_pending_keep) : std::nullopt;
    }

    // Preflight failed before any mutation. Consume the command and return to IDLE
    // with the last committed counters intact. Only a pending keep may be aborted:
    // a pending reset deliberately stays pending until the full prompt arrives.
    void abort_preflight() {
        OPENVINO_ASSERT(m_stage == Stage::PENDING && m_pending_keep > 0u,
                        "Continuous prefill: abort_preflight() without a pending keep command.");
        m_pending_keep = 0u;
        m_stage = Stage::IDLE;
    }

    // An exception after live state mutation began poisons the request. No new
    // count is published and only reset() can start recovery. Runs on exception
    // paths, so it must never throw.
    void poison() {
        m_pending_keep = 0u;
        m_stage = Stage::POISONED;
    }

    // Publish a committed prefill, full or continued. The new live count is also
    // the new prefill-produced watermark. Legal for an ordinary idle prefill or
    // a pending command that just executed; a commit must never clear the
    // poisoning that only reset() owns.
    void commit_prefill(uint32_t live_tokens) {
        OPENVINO_ASSERT(m_stage != Stage::POISONED, "Continuous prefill: commit_prefill() on a poisoned request.");
        m_live_tokens = live_tokens;
        m_watermark = live_tokens;
        m_pending_keep = 0u;
        m_stage = Stage::IDLE;
    }

    // Publish the result of an ordinary generate step. Generate-produced KV must
    // not advance the watermark; a speculative-decoding trim may even lower the
    // live count below it, in which case the watermark is clamped down. Generate
    // only runs when no command is pending, so anything but IDLE is an error.
    void publish_generate(uint32_t live_tokens) {
        OPENVINO_ASSERT(m_stage == Stage::IDLE, "Continuous prefill: publish_generate() outside the idle state.");
        m_live_tokens = live_tokens;
        m_watermark = std::min(m_watermark, live_tokens);
    }

    uint32_t live_tokens() const {
        return m_live_tokens;
    }

    uint32_t watermark() const {
        return m_watermark;
    }

    uint32_t chunk_size() const {
        return m_chunk_size;
    }

    uint32_t k_max() const {
        return m_k_max;
    }

private:
    bool m_enabled = false;
    Stage m_stage = Stage::IDLE;
    // Valid while PENDING: a positive value is a granted keep, zero is a reset.
    // The stage keeps "no command" distinct from "a command to keep 0".
    uint32_t m_pending_keep = 0u;
    uint32_t m_live_tokens = 0u;
    uint32_t m_watermark = 0u;
    uint32_t m_chunk_size = 0u;
    uint32_t m_k_max = 0u;
};

}  // namespace npuw
}  // namespace ov
