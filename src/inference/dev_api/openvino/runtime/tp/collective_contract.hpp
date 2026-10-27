// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Private collective contract between the orchestrator of tensor-parallel ranks (the TP plugin or an
 * external harness) and the device plugin that runs the collectives.
 *
 * Backend-neutral: no device types and no backend vocabulary. A backend is an opaque name the device plugin
 * defines and resolves; the only name defined here is "AUTO". Payload bodies are opaque bytes; only their common
 * header is defined here.
 *
 * @file openvino/runtime/tp/collective_contract.hpp
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace tp {

/// Opaque, implementation-defined payload.
using Blob = std::vector<uint8_t>;

/// The only backend name L3 defines; the device plugin resolves it to one of its own names.
inline constexpr const char* auto_backend = "AUTO";

/**
 * @brief One rank's endpoint of a collective group: a two-phase all-gather of opaque blobs, used at setup only.
 *
 * Every rank calls publish() and collect() alternately, the same number of times and in the same order.
 * publish() never blocks; collect() returns the blobs of all ranks for the round this rank published last,
 * indexed by rank, once every rank has published it, and throws on timeout. publish({}) followed by collect()
 * acts as a barrier.
 */
struct IExchange {
    virtual int rank() const = 0;
    virtual int world_size() const = 0;
    virtual void publish(const Blob& local) = 0;
    virtual std::vector<Blob> collect() = 0;
    virtual ~IExchange() = default;
};

namespace detail {

struct InProcessGroupState {
    std::mutex mutex;
    std::condition_variable cv;
    int world_size = 0;
    std::chrono::milliseconds timeout{0};
    struct Round {
        std::vector<Blob> blobs;  // indexed by rank
        int published = 0;
        int collected = 0;
    };
    std::map<uint64_t, Round> rounds;  // rounds in flight; erased once every rank has collected
};

class InProcessEndpoint final : public IExchange {
public:
    InProcessEndpoint(std::shared_ptr<InProcessGroupState> state, int rank) : m_state(std::move(state)), m_rank(rank) {}

    int rank() const override {
        return m_rank;
    }

    int world_size() const override {
        return m_state->world_size;
    }

    void publish(const Blob& local) override {
        OPENVINO_ASSERT(!m_published, "TP exchange: rank ", m_rank, " published twice without collecting");
        auto& s = *m_state;
        std::lock_guard<std::mutex> lock(s.mutex);
        auto& round = s.rounds[m_round];
        if (round.blobs.empty()) {
            round.blobs.resize(static_cast<size_t>(s.world_size));
        }
        round.blobs[static_cast<size_t>(m_rank)] = local;
        if (++round.published == s.world_size) {
            s.cv.notify_all();
        }
        m_published = true;
    }

    std::vector<Blob> collect() override {
        OPENVINO_ASSERT(m_published, "TP exchange: rank ", m_rank, " collects without having published");
        auto& s = *m_state;
        std::unique_lock<std::mutex> lock(s.mutex);
        // std::map references stay valid while other ranks add later rounds.
        auto& round = s.rounds[m_round];
        const auto all_published = [&] {
            return round.published == s.world_size;
        };
        if (s.timeout.count() == 0) {
            s.cv.wait(lock, all_published);
        } else {
            OPENVINO_ASSERT(s.cv.wait_for(lock, s.timeout, all_published),
                            "TP exchange timed out after ",
                            s.timeout.count(),
                            " ms: ",
                            round.published,
                            " of ",
                            s.world_size,
                            " ranks published");
        }
        std::vector<Blob> result = round.blobs;
        if (++round.collected == s.world_size) {
            s.rounds.erase(m_round);
        }
        ++m_round;
        m_published = false;
        return result;
    }

private:
    std::shared_ptr<InProcessGroupState> m_state;
    int m_rank;
    uint64_t m_round = 0;
    bool m_published = false;
};

}  // namespace detail

/// Reference IExchange for ranks that are threads of one process.
class InProcessExchange {
public:
    /// One shared group state and one endpoint per rank. A zero timeout waits indefinitely.
    static std::vector<std::shared_ptr<IExchange>> create_group(int world_size, std::chrono::milliseconds timeout) {
        OPENVINO_ASSERT(world_size >= 1, "TP exchange: world size must be positive, got ", world_size);
        OPENVINO_ASSERT(timeout.count() >= 0, "TP exchange: timeout must not be negative");
        auto state = std::make_shared<detail::InProcessGroupState>();
        state->world_size = world_size;
        state->timeout = timeout;
        std::vector<std::shared_ptr<IExchange>> endpoints;
        endpoints.reserve(static_cast<size_t>(world_size));
        for (int rank = 0; rank < world_size; ++rank) {
            endpoints.push_back(std::make_shared<detail::InProcessEndpoint>(state, rank));
        }
        return endpoints;
    }
};

/// Configuration of one collective group on one rank.
struct CollectiveGroup {
    std::shared_ptr<IExchange> exchange;  ///< this rank's endpoint of the group
    std::string backend = auto_backend;   ///< passed through unchanged from the user or harness
};

inline std::ostream& operator<<(std::ostream& os, const CollectiveGroup& group) {
    os << "{backend:" << group.backend;
    if (group.exchange) {
        os << ",rank:" << group.exchange->rank() << '/' << group.exchange->world_size();
    }
    return os << '}';
}

/**
 * @brief Common header of every exchanged payload (fixed size, little-endian).
 *
 * Carries the backend name the device plugin resolved ("AUTO" never appears here). Receivers validate the headers
 * of all ranks -- same magic, same resolved backend, a known and equal version -- before parsing the body.
 */
struct PayloadHeader {
    uint32_t magic = 0x5054564F;  // "OVTP"
    char backend[32] = {};        // resolved backend name, NUL-padded (at most 31 characters)
    uint32_t version = 0;         // version of the backend-specific body
};
static_assert(sizeof(PayloadHeader) == 40, "PayloadHeader is exchanged as raw bytes and must keep its layout");

namespace internal {

// ---- create_context() parameters ----

/// Devices of the TP group, e.g. {"GPU.0", ..., "GPU.N-1"}.
static constexpr ov::Property<std::vector<std::string>> device_group{"ov_internal_tp_device_group"};

/// group_id -> this rank's configuration of that group. Never serialized.
static constexpr ov::Property<std::map<int, CollectiveGroup>> groups{"ov_internal_tp_groups"};

/// Number of channels of the communicator (default 1): one per stream of the models bound to the context.
static constexpr ov::Property<uint32_t> channels{"ov_internal_tp_channels"};

/// Optional fixed size of the per-channel collective buffers; larger messages are chunked.
static constexpr ov::Property<size_t> comm_buffer_size{"ov_internal_tp_comm_buffer_size"};

/// Timeout of the setup rendezvous, the per-call rendezvous, and of collectives that do not complete.
static constexpr ov::Property<int> communication_timeout_ms{"COMMUNICATION_TIMEOUT_MS"};

// ---- compile_model() / import_model() config ----

/// Optional, default 0: stream s of the compiled model uses channel channel_base + s.
static constexpr ov::Property<uint32_t> channel_base{"ov_internal_tp_channel_base"};

// ---- Queries, arguments: device_group[, collective_backend] ----

/// Backend names the device plugin supports for the device group, in the order "AUTO" prefers them.
static constexpr ov::Property<std::vector<std::string>, ov::PropertyMutability::RO> collective_backends{
    "ov_internal_tp_collective_backends"};

/// Empty if the device group is supported with the backend, otherwise the reason. Does not throw for an
/// unsupported group.
static constexpr ov::Property<std::string, ov::PropertyMutability::RO> collective_support{
    "ov_internal_tp_collective_support"};

/// Argument of the support query (default "AUTO").
static constexpr ov::Property<std::string> collective_backend{"ov_internal_tp_collective_backend"};

}  // namespace internal
}  // namespace tp
}  // namespace ov
