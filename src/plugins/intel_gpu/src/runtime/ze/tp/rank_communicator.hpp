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

#include "collective_options.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"
#include "tp_device_coordinator.hpp"
#include "tp_l0_shared_context.hpp"

namespace cldnn {
namespace ze {
namespace tp {

/// Backend name this communicator resolves "AUTO" to and publishes in its payload header.
inline constexpr const char* l0_p2p_backend = "L0_P2P";

/// One rank's end of its group's collectives on one channel. A network bound to
/// a TP context holds one and calls its collectives through it.
class Channel {
public:
    Channel(TPDeviceCoordinatorPtr group, int rank) : m_group(std::move(group)), m_rank(rank) {}

    int rank() const {
        return m_rank;
    }
    int world_size() const {
        return m_group->world_size();
    }
    const CollectiveOptions& options() const {
        return m_group->options();
    }

    /// See TPDeviceCoordinator::run_spliced().
    bool run_spliced() const {
        return m_group->run_spliced();
    }

    /// See TPDeviceCoordinator::allreduce().
    void allreduce(int collective_id,
                   void* in_dev,
                   void* out_dev,
                   std::size_t n,
                   ov::element::Type dtype,
                   ze_kernel_handle_t reduce_kernel,
                   ze_command_list_handle_t model_queue) {
        m_group->allreduce(collective_id, m_rank, in_dev, out_dev, n, dtype, reduce_kernel, model_queue);
    }

    /// See TPDeviceCoordinator::gather_to_root().
    void gather_to_root(int collective_id,
                        void* in_dev,
                        void* out_dev,
                        std::size_t rows,
                        std::size_t slice_elems,
                        ov::element::Type dtype,
                        ze_command_list_handle_t model_queue) {
        m_group->gather_to_root(collective_id, m_rank, in_dev, out_dev, rows, slice_elems, dtype, model_queue);
    }

private:
    TPDeviceCoordinatorPtr m_group;
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
/// The first model bound to the context collects every rank's payload and
/// connects to the group state: the coordinator all ranks of the group share in
/// this process. Whichever rank binds first creates it; the others find it in a
/// process registry under the id of the group's rank-0 communicator.
///
/// One channel (one stream of one model) at a time.
class RankCommunicator {
public:
    RankCommunicator(std::shared_ptr<ov::tp::IExchange> exchange,
                     const std::string& backend,
                     TPL0SharedContextPtr shared,
                     std::chrono::milliseconds timeout,
                     const CollectiveOptions& options);

    RankCommunicator(const RankCommunicator&) = delete;
    RankCommunicator& operator=(const RankCommunicator&) = delete;

    int rank() const {
        return m_rank;
    }
    int world_size() const {
        return m_world_size;
    }
    const TPL0SharedContextPtr& shared_context() const {
        return m_shared;
    }
    const CollectiveOptions& options() const {
        return m_options;
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
    TPL0SharedContextPtr m_shared;
    std::chrono::milliseconds m_timeout;
    CollectiveOptions m_options;
    int m_rank;
    int m_world_size;
    uint64_t m_id;

    std::mutex m_mutex;
    TPDeviceCoordinatorPtr m_group;
    std::weak_ptr<Channel> m_channel;
};

using RankCommunicatorPtr = std::shared_ptr<RankCommunicator>;

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
