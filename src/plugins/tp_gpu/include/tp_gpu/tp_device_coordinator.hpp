// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/zero_api.hpp"

namespace ov {
namespace tp_gpu {

struct TPL0SharedContext;
using TPL0SharedContextPtr = std::shared_ptr<TPL0SharedContext>;

/// \brief Owns per-rank Level Zero resources and executes cross-GPU AllReduce
/// directly on device memory using a shared L0 context.
///
/// Lifetime: the coordinator must be created AFTER `TPL0SharedContext` and
/// destroyed BEFORE it.  All per-rank queues, command lists, modules and
/// kernels are scoped to the shared context.
///
/// Phase 2 scope:
///   * Build per-device kernel module from embedded OpenCL source via the
///     L0 OCLC extension.
///   * Provide a K=1 (single-chunk) AllReduce path: workers gather to main,
///     main reduces, main scatters to workers.  No dedicated copy queue.
///   * Plan is rebuilt whenever (rank-input ptr, rank-output ptr, n, dtype)
///     change for a collective; the previous plan is destroyed.
class TPDeviceCoordinator {
public:
    /// \param shared       Shared L0 context (driver + N devices + ze_context).
    /// \param world_size   Number of ranks (== shared->devices.size()).
    /// \param num_collectives  Number of distinct AllReduce points in the model.
    /// \param collective_timeout  Upper bound on any single wait inside a
    ///        collective, host- or device-side.  Zero means wait forever.
    TPDeviceCoordinator(TPL0SharedContextPtr shared,
                        int world_size,
                        int num_collectives,
                        std::chrono::milliseconds collective_timeout = std::chrono::milliseconds{5000});

    TPDeviceCoordinator(const TPDeviceCoordinator&) = delete;
    TPDeviceCoordinator& operator=(const TPDeviceCoordinator&) = delete;

    virtual ~TPDeviceCoordinator();

    /// Returns world size (number of ranks).
    int world_size() const { return m_world_size; }

    /// Returns true when initialization built kernels successfully on all ranks.
    bool is_ready() const { return m_ready; }

    /// True once a collective has failed.  The coordinator is single-use after
    /// that: the device queues may still hold unfinished work, so every later
    /// call throws instead of running on top of unknown state.
    bool is_aborted() const { return m_aborted.load(std::memory_order_acquire); }

    /// Marks the group as failed, wakes every waiting rank and makes all
    /// subsequent calls throw.  Safe to call from any rank; the first reason
    /// wins.  Must not be called while holding a rendezvous mutex.
    void abort_all(const std::string& reason);

    /// Coordinator-owned device-USM scratch accounting.  These allocations
    /// bypass the intel_gpu memory pool, so callers must account for them
    /// separately when attributing per-device VRAM.
    struct ScratchStats {
        std::size_t payload_capacity_bytes{0};
        std::size_t total_allocated_bytes{0};
        std::vector<std::size_t> allocated_bytes_per_rank;
        uint64_t generation{0};
        uint64_t growth_count{0};
        uint64_t allocation_count{0};
    };

    ScratchStats get_scratch_stats() const;

    /// In-place all-reduce-sum across all ranks.
    ///
    /// All N ranks must call this method concurrently (one thread per rank)
    /// for the same `collective_id` with consistent `n`/`dtype`.  On return
    /// each rank's `out_dev` holds the elementwise sum across all `in_dev`s.
    ///
    /// `in_dev` and `out_dev` may alias.  Both must be device USM allocated
    /// in the shared L0 context on the rank's device.
    ///
    /// Phase 2 supports `dtype` in {f16, f32}.
    virtual void allreduce(int collective_id,
                           int rank,
                           void* in_dev,
                           void* out_dev,
                           std::size_t n,
                           ov::element::Type dtype);

private:
    // Per-rank L0 state.
    struct RankState {
        ze_device_handle_t          device{nullptr};
        uint32_t                    compute_ordinal{0};
        // Regular path: queue + recordable cmdlist.  Immediate path:
        // compute_queue stays null and compute_list is created via
        // zeCommandListCreateImmediate (append-on-execute).
        ze_command_queue_handle_t   compute_queue{nullptr};
        ze_command_list_handle_t    compute_list{nullptr};

        // Optional dedicated copy engine.  When the device exposes a
        // copy-only queue group, we route the cross-device memcpy of the
        // allreduce onto it.  This offloads the bulk PCIe DMA from the
        // compute engine and typically lowers per-call submission latency
        // for small transfers and improves throughput for large ones.
        // When no dedicated copy engine exists, copy_queue/copy_list stay
        // null and the memcpy is recorded into compute_list as before.
        bool                        has_dedicated_copy{false};
        uint32_t                    copy_ordinal{0};
        ze_command_queue_handle_t   copy_queue{nullptr};
        ze_command_list_handle_t    copy_list{nullptr};

        ze_module_handle_t          module{nullptr};
        ze_kernel_handle_t          kernel_f16{nullptr};
        ze_kernel_handle_t          kernel_f32{nullptr};

        // Filled at init: GPU timer period (ns/tick) and the mask used to
        // wrap raw kernel-timestamp counter values.
        uint64_t                    timer_ns_per_tick{0};
        uint64_t                    timestamp_mask{~uint64_t{0}};

        // Persistent counter-based event used as the host-sync target on
        // the immediate path.  Counter-based events (Intel L0 extension)
        // are designed for host-sync on immediate cmdlists without the
        // queue-tracking round-trip incurred by zeCommandQueueSynchronize.
        // Null on the regular path.
        ze_event_handle_t           cb_event_done{nullptr};
    };

    // Cached plan for a single AllReduce collective.
    //
    // Per-call inputs that select which plan applies:
    //   collective_id (slot index), in_ptr, out_ptr, n, dtype
    struct Plan {
        // Signature
        std::vector<void*>          in_ptrs;        // [N]
        std::vector<void*>          out_ptrs;       // [N]
        std::size_t                 n{0};
        ov::element::Type           dtype{ov::element::dynamic};
        std::size_t                 max_payload_bytes{0};

        // Resources
        ze_event_pool_handle_t      pool{nullptr};
        std::vector<ze_event_handle_t> ev_recv;     // [N-1]
        ze_event_handle_t           ev_reduce{nullptr};
        std::vector<ze_event_handle_t> ev_bcast;    // [N-1]

        // Optional device-side timestamp probes (N=2 path).
        // ts_pool is a separate KERNEL_TIMESTAMP pool (timestamp events
        // require their own pool flag).  ev_ts_copy[r] is signaled by
        // rank r's memcpy; ev_ts_kernel[r] by rank r's reduce kernel.
        ze_event_pool_handle_t      ts_pool{nullptr};
        std::vector<ze_event_handle_t> ev_ts_copy;    // [N]
        std::vector<ze_event_handle_t> ev_ts_kernel;  // [N]

        bool matches(const std::vector<void*>& ins,
                     const std::vector<void*>& outs,
                     std::size_t want_n,
                     ov::element::Type want_dtype) const {
            return n == want_n && dtype == want_dtype && in_ptrs == ins && out_ptrs == outs;
        }
    };

    // Device-USM staging shared by all collective plans.  For TP=2 there
    // is one allocation on each rank device.  For N>2, allocations[0]
    // contains N-1 packed worker contributions on rank 0; other entries are
    // null.  Collectives are synchronous and outer InferRequests are
    // serialized, so only one plan uses the arena at a time.
    struct ScratchArena {
        std::vector<void*> allocations;              // [N]
        std::vector<std::size_t> bytes_per_rank;      // [N]
        std::size_t payload_capacity_bytes{0};
        std::size_t total_allocated_bytes{0};
        uint64_t generation{0};
        uint64_t growth_count{0};
        uint64_t allocation_count{0};
    };

    // Cross-rank rendezvous slot: collects each rank's (in,out) pointers for
    // a given collective_id before main rank records / executes.
    //
    // Three-stage barrier driven by generation counters:
    //   * enter:    arrived++; last-in flips enter_gen and resets arrived.
    //   * execute:  rank 0 sets done=true after launch + sync.
    //   * exit:     departed++; last-out flips exit_gen, resets departed,
    //               clears done + buffer pointers.
    // Generation counters avoid the races that arise from reusing a single
    // counter for both "rank 0 finished" and "all ranks have left".
    struct Rendezvous {
        std::mutex                  mtx;
        std::condition_variable     cv;
        std::vector<void*>          in_ptrs;        // [N], filled by ranks
        std::vector<void*>          out_ptrs;       // [N], filled by ranks
        int                         arrived{0};
        int                         departed{0};
        bool                        done{false};
        uint64_t                    enter_gen{0};
        uint64_t                    exit_gen{0};
        std::size_t                 n{0};
        ov::element::Type           dtype{ov::element::dynamic};
    };

    void init_rank(RankState& rs);
    void destroy_rank(RankState& rs);

    void destroy_plan(Plan& plan);
    void build_plan(int collective_id,
                    const std::vector<void*>& in_ptrs,
                    const std::vector<void*>& out_ptrs,
                    std::size_t n,
                    ov::element::Type dtype,
                    Plan& plan);
    void record_plan(Plan& plan);
    bool ensure_scratch_capacity(std::size_t payload_bytes);
    void destroy_scratch();
    void* scratch_buffer(int index) const;

    // Per-call host-side breakdown of execute_plan() (N=2 path).
    struct ExecStats {
        std::chrono::nanoseconds reset{};       // time spent resetting events
        std::chrono::nanoseconds submit{};      // both ExecuteCommandLists calls
        std::chrono::nanoseconds sync_a{};      // first  zeCommandQueueSynchronize
        std::chrono::nanoseconds sync_b{};      // second zeCommandQueueSynchronize
        // Device-side durations queried via kernel timestamps after sync.
        // Per rank we time the memcpy and the reduce kernel; aggregated as
        // max across ranks (slower side defines the critical path).
        std::chrono::nanoseconds dev_copy{};    // max(rank0, rank1) memcpy
        std::chrono::nanoseconds dev_kernel{};  // max(rank0, rank1) reduce kernel
        std::chrono::nanoseconds dev_ts_query{};// time spent in zeEventQueryKernelTimestamp
        // Bytes transferred per rank by the cross-device memcpy (n*elem_bytes).
        // On N=2 both ranks transfer the same amount; we report it once.
        std::size_t              copy_bytes{0};
    };

    void execute_plan(Plan& plan, ExecStats* stats = nullptr);

    /// Timeout translated to the nanosecond argument L0 sync calls take.
    /// Zero timeout maps to "no limit".
    uint64_t timeout_ns() const;

    /// zeCommandQueueSynchronize bounded by the collective timeout.  Throws on
    /// expiry so a stuck queue surfaces as an error instead of a frozen process.
    void sync_queue(ze_command_queue_handle_t queue, const char* what) const;

    /// Throws if a previous collective already failed.
    void throw_if_aborted() const;

    /// Aborts the group and throws.  `timed_out` distinguishes "this rank ran
    /// out of patience" from "somebody else already failed".
    [[noreturn]] void fail_collective(bool timed_out, int collective_id, int rank, const char* stage);

    TPL0SharedContextPtr            m_shared;
    int                             m_world_size{0};
    int                             m_num_collectives{0};
    bool                            m_ready{false};
    std::chrono::milliseconds       m_collective_timeout{5000};
    std::atomic<bool>               m_aborted{false};
    mutable std::mutex              m_abort_mutex;
    std::string                     m_abort_reason;
    // Toggle between regular-cmdlist + queue-sync (false) and
    // immediate-cmdlist + counter-based event sync (true).  Driven by
    // env var TP_USE_IMMEDIATE.  Latched at construction time.
    bool                            m_use_immediate{false};

    std::vector<RankState>          m_ranks;        // [N]

    mutable std::mutex              m_scratch_mutex;
    ScratchArena                    m_scratch;

    // Rank command lists are coordinator-wide rather than plan-owned.
    // Track which plan is physically recorded so alternating collective
    // IDs never execute a stale command list from another slot.
    Plan*                           m_recorded_plan{nullptr};
    uint64_t                        m_recorded_scratch_generation{0};

    // One rendezvous + one cached plan per collective_id.
    std::vector<std::unique_ptr<Rendezvous>> m_rendezvous;
    std::vector<std::unique_ptr<Plan>>       m_plans;
};

using TPDeviceCoordinatorPtr = std::shared_ptr<TPDeviceCoordinator>;

}  // namespace tp_gpu
}  // namespace ov
