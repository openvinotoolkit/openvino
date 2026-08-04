// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace ov {
namespace tp_gpu {

class TPDeviceCoordinator;

/// \brief Reusable barrier for N threads (C++17 compatible).
class SimpleBarrier {
public:
    explicit SimpleBarrier(int count) : m_count(count), m_waiting(0), m_generation(0) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(m_mutex);
        int gen = m_generation;
        if (++m_waiting == m_count) {
            m_waiting = 0;
            ++m_generation;
            m_cv.notify_all();
        } else {
            m_cv.wait(lock, [this, gen] { return gen != m_generation; });
        }
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    int m_count;
    int m_waiting;
    int m_generation;
};

/// \brief Shared coordination state for cross-GPU AllReduce.
///
/// One instance is created per compiled TP model and shared between all ranks.
/// Each collective AllReduce point gets a Collective slot with:
///   - per-rank staging buffers
///   - a barrier for synchronizing ranks
///
/// Thread model: each rank runs on its own std::async thread. The barrier
/// ensures all ranks have deposited their data before any rank reads.
class TPCoordination {
public:
    TPCoordination(int world_size, int num_collectives)
        : m_world_size(world_size) {
        m_collectives.reserve(num_collectives);
        for (int i = 0; i < num_collectives; ++i) {
            m_collectives.push_back(std::make_unique<Collective>(world_size));
        }
    }

    /// Deposit this rank's partial-sum data into the staging buffer.
    void deposit(int collective_id, int rank, const void* data, size_t bytes) {
        auto& coll = *m_collectives[collective_id];
        auto& buf = coll.rank_buffers[rank];
        if (buf.size() < bytes) {
            buf.resize(bytes);
        }
        std::memcpy(buf.data(), data, bytes);
    }

    /// Barrier: wait until all ranks have deposited.
    void barrier_wait(int collective_id) {
        m_collectives[collective_id]->sync.arrive_and_wait();
    }

    /// Get a pointer to rank's deposited data.
    const void* get_buffer(int collective_id, int rank) const {
        return m_collectives[collective_id]->rank_buffers[rank].data();
    }

    int world_size() const { return m_world_size; }

    /// Optional device-side AllReduce coordinator. When set and the
    /// `tp_allreduce` primitive's input/output reside in USM-device
    /// memory of the rank's GPU, the primitive bypasses the CPU stage
    /// and dispatches the collective directly through this coordinator.
    void set_device_coordinator(std::shared_ptr<TPDeviceCoordinator> dc) {
        m_device_coordinator = std::move(dc);
    }
    const std::shared_ptr<TPDeviceCoordinator>& device_coordinator() const {
        return m_device_coordinator;
    }

private:
    struct Collective {
        explicit Collective(int world_size)
            : rank_buffers(world_size),
              sync(world_size) {}

        Collective(const Collective&) = delete;
        Collective& operator=(const Collective&) = delete;

        std::vector<std::vector<uint8_t>> rank_buffers;
        SimpleBarrier sync;
    };

    int m_world_size;
    std::vector<std::unique_ptr<Collective>> m_collectives;
    std::shared_ptr<TPDeviceCoordinator> m_device_coordinator;
};

}  // namespace tp_gpu
}  // namespace ov
