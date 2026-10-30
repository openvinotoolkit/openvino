// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "collective_backend.hpp"
#include "collective_options.hpp"
#include "group_state.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"
#include "tp_l0_shared_context.hpp"

namespace cldnn {
namespace ze {
namespace tp {

/// One rank's end of its group's collective resources on one channel. A network
/// bound to a TP context holds one; its collective implementations run over it.
class Channel {
public:
    Channel(GroupStatePtr group, int rank) : m_group(std::move(group)), m_rank(rank) {}

    int rank() const {
        return m_rank;
    }
    int world_size() const {
        return m_group->world_size();
    }
    const CollectiveOptions& options() const {
        return m_group->options();
    }
    bool run_spliced() const {
        return m_group->run_spliced();
    }
    const GroupStatePtr& group() const {
        return m_group;
    }
    /// Throws if the group has failed: whatever the devices produced since may be garbage.
    void throw_if_aborted() const {
        m_group->throw_if_aborted();
    }

private:
    GroupStatePtr m_group;
    int m_rank;
};

/// A collective of a model, as far as binding it to a communicator is concerned.
struct CollectiveUse {
    std::string name;
    uint32_t collective_id = 0;
    uint32_t rank = 0;
    uint32_t world_size = 0;
};

/// \brief The TP communicator of one rank, owned by that rank's GPU remote context.
///
/// Created with the context, it publishes this rank's setup payload through the
/// group's IExchange endpoint and returns without waiting for the other ranks.
/// With a fixed staging size it also allocates the rank's staging then, so the
/// memory is taken before any model is compiled.
/// The first model bound to the context collects every rank's payload and
/// connects to the group state all ranks of the group share in this process. Whichever rank binds first creates it; the others find it in a
/// process registry under the id of the group's rank-0 communicator.
///
/// One channel (one stream of one model) at a time.
class RankCommunicator {
public:
    /// \param backend  The resolved collective backend (resolve_backend()),
    ///        published in the payload header; every rank must have the same.
    /// \param comm_buffer_size  Bytes of this rank's staging, fixed for the
    ///        context's lifetime; larger messages are chunked. Zero lets the
    ///        staging grow with the largest message. Equal on every rank.
    RankCommunicator(std::shared_ptr<ov::tp::IExchange> exchange,
                     const std::string& backend,
                     TPL0SharedContextPtr shared,
                     std::chrono::milliseconds timeout,
                     const CollectiveOptions& options,
                     std::size_t comm_buffer_size = 0);

    RankCommunicator(const RankCommunicator&) = delete;
    RankCommunicator& operator=(const RankCommunicator&) = delete;

    int rank() const {
        return m_rank;
    }
    int world_size() const {
        return m_world_size;
    }
    const std::string& backend() const {
        return m_backend;
    }
    const TPL0SharedContextPtr& shared_context() const {
        return m_shared;
    }
    const CollectiveOptions& options() const {
        return m_options;
    }

    /// The engine of this rank's context; the rank's collective staging is
    /// counted in its memory statistics. Before the first bind.
    void set_memory_account(std::weak_ptr<cldnn::engine> engine) {
        m_memory_account = std::move(engine);
    }

    /// Binds a model's collectives to channel `channel` of this communicator.
    ///
    /// Checks every collective against this rank and group, connects to the
    /// group state on the first call, and returns what the model's network runs
    /// its collectives through. Null for a model without collectives.
    std::shared_ptr<Channel> bind(const std::vector<CollectiveUse>& collectives, uint32_t channel);

private:
    /// Collects the payloads, validates them and finds or creates the group state.
    void connect(int num_collectives);

    std::shared_ptr<ov::tp::IExchange> m_exchange;
    std::string m_backend;
    TPL0SharedContextPtr m_shared;
    std::chrono::milliseconds m_timeout;
    CollectiveOptions m_options;
    int m_rank;
    int m_world_size;
    uint64_t m_id;
    std::size_t m_staging_bytes{0};
    std::shared_ptr<void> m_staging;
    std::weak_ptr<cldnn::engine> m_memory_account;

    std::mutex m_mutex;
    GroupStatePtr m_group;
    std::weak_ptr<Channel> m_channel;
};

using RankCommunicatorPtr = std::shared_ptr<RankCommunicator>;

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
