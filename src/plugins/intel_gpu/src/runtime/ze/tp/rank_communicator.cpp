// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rank_communicator.hpp"

#include "intel_gpu/runtime/itt.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace cldnn {
namespace ze {
namespace tp {

namespace {

constexpr uint32_t payload_version = 2;

/// What every rank publishes after the common header. Only meaningful inside
/// one process: the ranks find their shared state by the rank-0 id.
struct PayloadBody {
    uint64_t pid = 0;
    uint64_t communicator_id = 0;
    uint32_t rank = 0;
    uint32_t world_size = 0;
    uint64_t staging_bytes = 0;
};

uint64_t current_pid() {
#ifdef _WIN32
    return static_cast<uint64_t>(_getpid());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

uint64_t next_communicator_id() {
    static std::atomic<uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

ov::tp::Blob make_payload(const std::string& backend, const PayloadBody& body) {
    ov::tp::PayloadHeader header;
    std::strncpy(header.backend, backend.c_str(), sizeof(header.backend) - 1);
    header.version = payload_version;

    ov::tp::Blob blob(sizeof(header) + sizeof(body));
    std::memcpy(blob.data(), &header, sizeof(header));
    std::memcpy(blob.data() + sizeof(header), &body, sizeof(body));
    return blob;
}

PayloadBody parse_payload(const ov::tp::Blob& blob, int from_rank, const std::string& own_backend) {
    OPENVINO_ASSERT(blob.size() == sizeof(ov::tp::PayloadHeader) + sizeof(PayloadBody),
                    "[GPU][TP] rank ", from_rank, " published a payload of ", blob.size(), " bytes, expected ",
                    sizeof(ov::tp::PayloadHeader) + sizeof(PayloadBody));

    ov::tp::PayloadHeader header;
    std::memcpy(&header, blob.data(), sizeof(header));
    const ov::tp::PayloadHeader expected;
    OPENVINO_ASSERT(header.magic == expected.magic, "[GPU][TP] rank ", from_rank, " published a foreign payload");
    const std::string backend(header.backend, strnlen(header.backend, sizeof(header.backend)));
    OPENVINO_ASSERT(backend == own_backend,
                    "[GPU][TP] rank ", from_rank, " resolved collective backend \"", backend, "\", this rank \"",
                    own_backend, "\"");
    OPENVINO_ASSERT(header.version == payload_version,
                    "[GPU][TP] rank ", from_rank, " published payload version ", header.version, ", this rank ",
                    payload_version);

    PayloadBody body;
    std::memcpy(&body, blob.data() + sizeof(header), sizeof(body));
    return body;
}

/// The group states of this process, by the id of their rank-0 communicator.
/// An id rather than an address: a communicator's address can be reused while
/// the state of its group is still held by the other ranks.
std::mutex registry_mutex;
std::map<uint64_t, std::weak_ptr<GroupState>> registry;

}  // namespace

RankCommunicator::RankCommunicator(std::shared_ptr<ov::tp::IExchange> exchange,
                                   const std::string& backend,
                                   TPL0SharedContextPtr shared,
                                   std::chrono::milliseconds timeout,
                                   const CollectiveOptions& options,
                                   std::size_t comm_buffer_size)
    : m_exchange(std::move(exchange)),
      m_backend(backend),
      m_shared(std::move(shared)),
      m_timeout(timeout),
      m_options(options),
      m_rank(0),
      m_world_size(0),
      m_id(next_communicator_id()),
      // Two halves, each a multiple of 256 bytes.
      m_staging_bytes(comm_buffer_size / 512 * 512) {
    OPENVINO_ASSERT(m_exchange != nullptr, "[GPU][TP] the collective group has no exchange endpoint");
    OPENVINO_ASSERT(m_shared != nullptr, "[GPU][TP] the communicator needs the device-group context");
    m_rank = m_exchange->rank();
    m_world_size = m_exchange->world_size();
    OPENVINO_ASSERT(m_world_size >= 2 && m_rank >= 0 && m_rank < m_world_size,
                    "[GPU][TP] invalid rank ", m_rank, " of ", m_world_size);
    OPENVINO_ASSERT(m_shared->devices.size() == static_cast<size_t>(m_world_size),
                    "[GPU][TP] the device group has ", m_shared->devices.size(), " devices for ", m_world_size,
                    " ranks");
    OPENVINO_ASSERT(m_backend == l0_p2p_backend,
                    "[GPU][TP] the communicator runs collective backend ", l0_p2p_backend, ", got \"", m_backend,
                    "\"");
    OPENVINO_ASSERT(comm_buffer_size == 0 || m_staging_bytes != 0,
                    "[GPU][TP] ", ov::tp::internal::comm_buffer_size.name(), " of ", comm_buffer_size,
                    " bytes is too small");
    if (m_staging_bytes != 0) {
        m_staging = GroupState::allocate_staging(m_shared, m_rank, m_staging_bytes);
    }

    PayloadBody body;
    body.pid = current_pid();
    body.communicator_id = m_id;
    body.rank = static_cast<uint32_t>(m_rank);
    body.world_size = static_cast<uint32_t>(m_world_size);
    body.staging_bytes = m_staging_bytes;
    m_exchange->publish(make_payload(m_backend, body));
}

void RankCommunicator::connect(int num_collectives) {
    OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_communicator_connect");
    const auto blobs = m_exchange->collect();
    OPENVINO_ASSERT(blobs.size() == static_cast<size_t>(m_world_size),
                    "[GPU][TP] collected ", blobs.size(), " payloads for ", m_world_size, " ranks");

    const auto pid = current_pid();
    std::vector<PayloadBody> bodies;
    for (int r = 0; r < m_world_size; ++r) {
        const auto body = parse_payload(blobs[static_cast<size_t>(r)], r, m_backend);
        OPENVINO_ASSERT(body.rank == static_cast<uint32_t>(r) && body.world_size == static_cast<uint32_t>(m_world_size),
                        "[GPU][TP] payload ", r, " claims rank ", body.rank, " of ", body.world_size);
        OPENVINO_ASSERT(body.pid == pid,
                        "[GPU][TP] rank ", r, " runs in another process; ranks of one group must share a process");
        // Chunk sizes follow from the staging size, and the ranks must agree on them.
        OPENVINO_ASSERT(body.staging_bytes == m_staging_bytes,
                        "[GPU][TP] rank ", r, " has ", body.staging_bytes, " bytes of fixed staging, rank ", m_rank,
                        " ", m_staging_bytes, "; ", ov::tp::internal::comm_buffer_size.name(),
                        " must be the same on every rank");
        bodies.push_back(body);
    }
    OPENVINO_ASSERT(bodies[static_cast<size_t>(m_rank)].communicator_id == m_id,
                    "[GPU][TP] the payload at this rank's index belongs to another communicator");

    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto key = bodies.front().communicator_id;
    auto group = registry[key].lock();
    if (!group) {
        group = std::make_shared<GroupState>(m_shared, m_world_size, num_collectives, m_timeout, m_options,
                                             m_staging_bytes);
        for (auto it = registry.begin(); it != registry.end();) {
            it = it->second.expired() ? registry.erase(it) : std::next(it);
        }
        registry[key] = group;
    }
    OPENVINO_ASSERT(group->shared_context() == m_shared,
                    "[GPU][TP] the ranks of one group were created over different device groups");
    OPENVINO_ASSERT(group->num_collectives() == num_collectives,
                    "[GPU][TP] rank ", m_rank, " has ", num_collectives, " collectives, the group was set up for ",
                    group->num_collectives());
    m_group = std::move(group);
}

std::shared_ptr<Channel> RankCommunicator::bind(const std::vector<CollectiveUse>& collectives, uint32_t channel) {
    if (collectives.empty()) {
        return nullptr;
    }
    OPENVINO_ASSERT(channel == 0,
                    "[GPU][TP] the TP context has one channel, stream ", channel,
                    " cannot be bound; compile the rank model with a single stream");

    std::vector<bool> seen(collectives.size(), false);
    for (const auto& use : collectives) {
        OPENVINO_ASSERT(use.rank == static_cast<uint32_t>(m_rank) && use.world_size == static_cast<uint32_t>(m_world_size),
                        "[GPU][TP] collective ", use.name, " is for rank ", use.rank, " of ", use.world_size,
                        ", the context is rank ", m_rank, " of ", m_world_size);
        OPENVINO_ASSERT(use.collective_id < collectives.size() && !seen[use.collective_id],
                        "[GPU][TP] collective ", use.name, " has id ", use.collective_id,
                        "; ids must number the model's ", collectives.size(), " collectives once each");
        seen[use.collective_id] = true;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(m_channel.expired(), "[GPU][TP] channel ", channel, " of the TP context is already bound to a model");
    const auto count = static_cast<int>(collectives.size());
    if (!m_group) {
        connect(count);
        m_group->set_memory_account(m_rank, m_memory_account);
        if (m_staging) {
            m_group->adopt_staging(m_rank, m_staging);
        }
    }
    OPENVINO_ASSERT(m_group->num_collectives() == count,
                    "[GPU][TP] the model has ", count, " collectives, the context's group was set up for ",
                    m_group->num_collectives());
    // A failed group is terminal: the ranks' call counters may differ.
    m_group->throw_if_aborted();

    auto bound = std::make_shared<Channel>(m_group, m_rank);
    m_channel = bound;
    return bound;
}

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
