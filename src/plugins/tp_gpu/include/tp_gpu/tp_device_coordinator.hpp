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
#include <thread>
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

    /// Number of distinct AllReduce points the coordinator was sized for.
    /// Recorded in the compiled blob so an imported model can rebuild an
    /// identically sized coordinator without re-analyzing the graph.
    int num_collectives() const { return m_num_collectives; }

    /// Returns true when initialization built kernels successfully on all ranks.
    bool is_ready() const { return m_ready; }

    /// True once a collective has failed.  The coordinator is single-use after
    /// that: the device queues may still hold unfinished work, so every later
    /// call throws instead of running on top of unknown state.
    bool is_aborted() const { return m_aborted.load(std::memory_order_acquire); }

    /// Tells the watchdog that this rank has handed a collective to a queue
    /// whose completion the host will not wait for.
    ///
    /// Draining our own queue used to be what caught a dead peer: a rank that
    /// never signalled left `zeCommandQueueSynchronize` to time out.  Once the
    /// collective rides in the model's queue there is no such call -- the wait
    /// happens on the device, where Level Zero offers no deadline at all, and
    /// a rank that dies takes the whole inference down into a hang.
    void note_collective_started(int rank);

    /// Hands this rank's share of an AllReduce to `model_queue` -- the
    /// immediate command list intel_gpu already runs the model on -- and
    /// returns without waiting for it.
    ///
    /// The queue is in-order, so the recording lands after the operations
    /// that produced `in_dev` and before whatever reads `out_dev`, with no
    /// drain in between.  That is the whole point: the host goes on
    /// dispatching the next stretch of the model while four devices work
    /// through the collective, instead of stopping at every one of them.
    ///
    /// Falls back to the synchronous path when the driver has no splice.
    ///
    /// Virtual for the same reason allreduce() is: the GPU plugin calls the
    /// coordinator through this vtable without linking against the plugin
    /// that defines it.
    virtual void allreduce_async(int collective_id,
                                 int rank,
                                 void* in_dev,
                                 void* out_dev,
                                 std::size_t n,
                                 ov::element::Type dtype,
                                 ze_command_list_handle_t model_queue);

    /// Whether allreduce_async can do anything but forward to allreduce().
    virtual bool async_supported() const;


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

    /// Collects each rank's slice of a row-major matrix into rank 0's buffer.
    ///
    /// Every rank holds `rows` x `slice_elems` elements of its own; rank 0's
    /// buffer is `rows` x (`slice_elems` * world_size), and rank r's slice
    /// belongs at column offset r * slice_elems of every row -- so the slices
    /// are strided in the destination, not contiguous.
    ///
    /// Only rank 0's buffer is written.  That is what the vocabulary
    /// projection needs: the infer request reads outputs from rank 0 alone,
    /// so gathering there instead of to everyone saves world_size-1 transfers
    /// per rank for a result nobody would look at.
    ///
    /// `out_dev` is the destination on rank 0 and ignored on every other rank.
    /// Blocking, like allreduce: returns once every slice has landed.
    virtual void gather_to_root(int collective_id,
                                int rank,
                                void* in_dev,
                                void* out_dev,
                                std::size_t rows,
                                std::size_t slice_elems,
                                ov::element::Type dtype);

    /// gather_to_root spliced into the model's queue, the counterpart of
    /// allreduce_async.
    ///
    /// The blocking form is the expensive one here: it drains the rank's
    /// model stream, runs the copy on the coordinator's own queue and waits
    /// for it, once per token on the vocabulary projection.  Splicing removes
    /// both waits, but it also removes what made the blocking form correct --
    /// the host barrier no longer implies the slices have landed, because the
    /// ranks now write from four independent queues.  The recording carries
    /// that ordering instead: every other rank signals when its slice is
    /// down, and the root's recording waits on all of them, so the model
    /// queue of the rank that reads the gathered buffer is held until it is
    /// whole.
    ///
    /// Falls back to the synchronous path when the driver has no splice.
    virtual void gather_to_root_async(int collective_id,
                                      int rank,
                                      void* in_dev,
                                      void* out_dev,
                                      std::size_t rows,
                                      std::size_t slice_elems,
                                      ov::element::Type dtype,
                                      ze_command_list_handle_t model_queue);

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
        // What the recorded lists do.  A collective id belongs to exactly one
        // of these for the life of the model, so the two share the same slot,
        // the same command lists and the same rendezvous.
        enum class Kind { allreduce, gather };
        Kind                        kind{Kind::allreduce};

        // Which of the coordinator's resource sets this plan draws on.  Baked
        // into the recording, because the staging addresses are.
        int                         buffer{0};

        // Gather only: rows of the matrix being collected, and how many
        // elements of each row this rank contributes.  `n` stays the total
        // element count of the local slice so the signature check is uniform.
        std::size_t                 rows{0};
        std::size_t                 slice_elems{0};
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

        // Signalled when this rank's spliced recording has finished on the
        // device.  Host-visible, unlike everything else here, because two
        // parties ask about it from the host: the rank itself, before
        // splicing the same list again, and the watchdog, which has no other
        // way to tell a device that is merely behind from one that is stuck.
        ze_event_pool_handle_t      done_pool{nullptr};
        std::vector<ze_event_handle_t> ev_done;     // [N]
        // Whether that event belongs to a splice that has not been accounted
        // for yet.  Atomic because the watchdog reads it while the ranks
        // write it.
        std::unique_ptr<std::atomic<uint8_t>[]> in_flight;  // [N]

        // Ring schedule: one event per (step, rank), signaled by the rank's
        // outgoing copy at that step and waited on by its successor.  Laid
        // out as [step * N + rank] over 2*(N-1) steps.
        std::vector<ze_event_handle_t> ev_ring;

        // Gather only: rank r != 0 signals ev_gather[r] when its slice is
        // down, and the root waits on all of them.  Created on first use --
        // one collective out of the whole model is a gather, so building
        // these for every plan would be waste.
        ze_event_pool_handle_t      gather_pool{nullptr};
        std::vector<ze_event_handle_t> ev_gather;   // [N], index 0 unused

        // Optional device-side timestamp probes (N=2 path).
        // ts_pool is a separate KERNEL_TIMESTAMP pool (timestamp events
        // require their own pool flag).  ev_ts_copy[r] is signaled by
        // rank r's memcpy; ev_ts_kernel[r] by rank r's reduce kernel.
        ze_event_pool_handle_t      ts_pool{nullptr};
        std::vector<ze_event_handle_t> ev_ts_copy;    // [N]
        std::vector<ze_event_handle_t> ev_ts_kernel;  // [N]

        // Command lists holding this collective's recorded commands, one per
        // rank.  Every collective owns its own so that a recording survives
        // the next collective: with a single list per rank each call
        // overwrote the previous recording, and one decode step re-recorded
        // all 64 collectives for nothing.  Empty on the immediate path,
        // which appends directly at execute time.
        std::vector<ze_command_list_handle_t> compute_lists;  // [N]
        std::vector<ze_command_list_handle_t> copy_lists;     // [N], null without a copy engine

        // Per rank: whether that rank's lists hold commands matching the
        // signature above, and which scratch generation they were recorded
        // against (the staging pointers are baked into the recording).
        // Per rank rather than per plan because ring and the direct N=2
        // exchange record each rank from that rank's own pointers, so a rank
        // can be re-recorded without touching the others.  uint8_t rather
        // than bool: std::vector<bool> packs bits, and neighbouring ranks
        // would then be writing the same word.
        std::vector<uint8_t>        recorded;                     // [N]
        std::vector<uint64_t>       recorded_scratch_generation;  // [N]

        bool matches(const std::vector<void*>& ins,
                     const std::vector<void*>& outs,
                     std::size_t want_n,
                     ov::element::Type want_dtype) const {
            return n == want_n && dtype == want_dtype && in_ptrs == ins && out_ptrs == outs;
        }

        bool matches_gather(const std::vector<void*>& ins,
                            const std::vector<void*>& outs,
                            std::size_t want_rows,
                            std::size_t want_slice,
                            ov::element::Type want_dtype) const {
            return kind == Kind::gather && rows == want_rows && slice_elems == want_slice &&
                   dtype == want_dtype && in_ptrs == ins && out_ptrs == outs;
        }
    };

    // Device-USM staging shared by all collective plans.
    //
    //   * TP=2 direct exchange: one payload-sized allocation per rank.
    //   * Ring (N>2): N chunk slots per rank, so every step of the
    //     reduce-scatter lands in a slot of its own and no cross-rank
    //     back-pressure is needed between steps.  Total per rank is one
    //     payload, the same order as TP=2 -- unlike the funnel, which piled
    //     (N-1) payloads onto rank 0 alone.
    //   * Funnel (N>2 fallback): allocations[0] holds N-1 packed worker
    //     contributions on rank 0; other entries are null.
    //
    // Collectives are synchronous and outer InferRequests are serialized, so
    // only one plan uses the arena at a time.
    struct ScratchArena {
        std::vector<void*> allocations;              // [N]
        std::vector<std::size_t> bytes_per_rank;      // [N]
        std::size_t payload_capacity_bytes{0};
        std::size_t chunk_capacity_bytes{0};          // ring: stride between slots
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
    //
    // A mutex and a condition_variable, deliberately.  These barriers are
    // crossed twice per collective and 64 collectives per model step, which
    // looks like an obvious case for a lock-free counter -- but an atomic
    // barrier that spins before parking measured worse at four ranks (26.0 vs
    // 24.3 ms/token), and worse still when the fallback slept instead of
    // spinning (28.9).  What a rank waits for here is mostly its peers being
    // genuinely late, not the barrier itself, and a spinning waiter takes the
    // core that the late peer's dispatch thread needs.  futex parks and wakes
    // better than anything hand-rolled here did.
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
        // When the first rank reached this collective, used to measure how
        // far behind the others are.  Only written under TP_SKEW.
        std::chrono::steady_clock::time_point first_arrival{};
    };

    // How far apart the ranks arrive at a collective, and what each of them
    // spends its time waiting for.  Collected only when TP_SKEW is set, and
    // deliberately separate from TP_PROF: profiling adds hundreds of
    // microseconds per collective and would drown the imbalance being
    // measured.  Every counter is indexed by rank and written only by that
    // rank, except the two totals, which are written under the rendezvous
    // mutex by whichever rank arrives last.
    struct SkewStats {
        std::vector<uint64_t> ph1_ns;      // [N] time spent in the enter barrier
        std::vector<uint64_t> ph2_ns;      // [N] time spent in the execute phase
        std::vector<uint64_t> ph3_ns;      // [N] time spent in the exit barrier
        std::vector<uint64_t> late_ns;     // [N] arrival minus the first arrival
        std::vector<uint64_t> last_count;  // [N] how often this rank arrived last
        // The stretch of model work between leaving one collective and
        // reaching the next.  This is where the arrival skew is built, so it
        // is measured per rank with its extremes kept, not just averaged.
        std::vector<uint64_t> seg_ns;      // [N] sum of segment durations
        std::vector<uint64_t> seg_min_ns;  // [N]
        std::vector<uint64_t> seg_max_ns;  // [N]
        std::vector<uint64_t> seg_count;   // [N]
        std::vector<std::chrono::steady_clock::time_point> last_exit;  // [N]
        // Breakdown of ph2 on the splice path.  The phase looked far more
        // expensive than the 1.3 us the append itself costs, and the pieces
        // are not guessable from the outside: the gate is a condvar handoff
        // for every rank but rank 0, and the event wait is only supposed to
        // fire when the host has run two collectives ahead of the device.
        std::vector<uint64_t> p2_gate_ns;    // [N] rank 0: prep; others: wait for `done`
        std::vector<uint64_t> p2_rec_ns;     // [N] recording this rank's commands
        std::vector<uint64_t> p2_wait_ns;    // [N] zeEventHostSynchronize on the previous splice
        std::vector<uint64_t> p2_reset_ns;   // [N] zeEventHostReset
        std::vector<uint64_t> p2_append_ns;  // [N] the splice itself
        std::vector<uint64_t> p2_rec_count;  // [N] how often a re-record happened
        std::vector<uint64_t> p2_wait_count; // [N] how often the previous splice was still in flight
        std::vector<uint64_t> p2_block_count;// [N] how often that wait actually had to block
        // How long the spliced collective occupied the rank's queue, read off
        // the completion event's kernel timestamps.  This is the one part of
        // the cost that no host phase contains: the splice hands the work over
        // and returns, so the copies, the reduce kernel and the waits on peers
        // all happen after every host measurement has ended.  Sampled where
        // the previous splice is already being queried anyway, so it costs one
        // extra call per collective and no synchronization.
        std::vector<uint64_t> p2_dev_ns;     // [N] sum of (kernelEnd - kernelStart)
        std::vector<uint64_t> p2_dev_max_ns; // [N]
        std::vector<uint64_t> p2_dev_count;  // [N]
        uint64_t spread_ns{0};             // sum of (last arrival - first arrival)
        uint64_t calls{0};
    };
    SkewStats                       m_skew;

    void init_rank(RankState& rs);
    void destroy_rank(RankState& rs);

    void destroy_plan(Plan& plan);

    /// Creates the plan's event pool and its events.  How many there are
    /// depends only on the world size and on whether profiling is on, never on
    /// the payload, so this runs once at construction.  Doing it lazily put 64
    /// pool creations on the first inference, which is the one whose latency
    /// users measure as time to first token.
    void create_plan_events(Plan& plan);
    void record_plan(Plan& plan);

    /// True when the schedule records every rank from that rank's own
    /// pointers plus the coordinator's staging, and so can be driven one rank
    /// at a time.  The funnel cannot: it writes into peer output buffers, so
    /// recording it needs every rank's pointers at once.
    bool per_rank_schedule() const { return m_use_ring || m_world_size == 2; }

    /// Drains, resets and re-records one rank's lists.  Only valid for a
    /// per-rank schedule.
    void record_rank(Plan& plan, int rank);

    /// Submits one rank's lists.  Every rank must be submitted before any of
    /// them is waited on: a rank's list blocks on events its neighbours only
    /// signal once they run, so draining one first would deadlock.
    void submit_rank(Plan& plan, int rank);

    /// Waits for one rank's queues to drain.
    void sync_rank(Plan& plan, int rank);

    /// Records the ring schedule for one rank: N-1 reduce-scatter steps
    /// followed by N-1 all-gather steps, sending only to its successor.  Each
    /// link carries 2*(N-1)/N of the payload instead of the funnel's (N-1)
    /// copies in and out of rank 0, so the cost per link stops growing with N.
    void record_ring_rank(Plan& plan, int rank);

    /// Records one rank's contribution to a gather: a single strided copy of
    /// its slice into the root's buffer.  Ranks write disjoint columns, so
    /// there is no data to order between them, but the root still has to know
    /// when the columns are down: every other rank signals ev_gather[r] and
    /// the root's recording waits on all of them before it ends.  On the
    /// spliced path that wait is the only thing holding the root's model
    /// queue back; on the blocking path the exit barrier would do as well,
    /// and one recording shape for both keeps the two from drifting apart.
    void record_gather_rank(Plan& plan, int rank);

    /// Creates the gather ordering events on `plan` if they are not there.
    void ensure_gather_events(Plan& plan);

    /// Waits until the previous splice of this rank's recording has finished,
    /// so the list can be appended or re-recorded.  The allreduce path has
    /// the same wait inlined, where it also feeds the skew counters.
    void await_previous_splice(Plan& plan, int rank, int collective_id);

    /// The body behind gather_to_root() and gather_to_root_async().
    void run_gather(int collective_id,
                    int rank,
                    void* in_dev,
                    void* out_dev,
                    std::size_t rows,
                    std::size_t slice_elems,
                    ov::element::Type dtype,
                    ze_command_list_handle_t model_queue);

    /// Records the direct two-rank exchange for one rank: push our input into
    /// the peer's staging, then reduce our input with what the peer pushed
    /// into ours.
    void record_pair_rank(Plan& plan, int rank);

    /// Element range of ring chunk `chunk` within a payload of `n` elements.
    /// Chunks differ by at most one element, which keeps every one of them
    /// non-empty and removes the need to signal skipped steps.
    void ring_chunk(std::size_t n, int chunk, std::size_t& offset, std::size_t& count) const;

    /// Staging slot for `chunk` on `rank` inside the ring arena, in the
    /// half of it belonging to `buffer`.
    void* ring_slot(int rank, int chunk, int buffer) const;

    /// How many independent sets of plans and staging exist per collective.
    ///
    /// Two, once ranks run their own schedule: a rank that has left a
    /// collective may reach the same one again on the next token while its
    /// neighbours are still executing the previous instance, and the ring
    /// writes into a neighbour's staging.  Alternating between two sets means
    /// the recording being laid down and the one still running never share a
    /// command list, an event or a byte of the arena.  The funnel keeps one
    /// set: rank 0 drives it from a single thread and nothing overlaps.
    int plan_buffers() const { return per_rank_schedule() ? 2 : 1; }

    /// The plan holding `collective_id` in the given buffer.
    Plan& plan_at(int collective_id, int buffer) const {
        return *m_plans[static_cast<std::size_t>(collective_id) *
                            static_cast<std::size_t>(plan_buffers()) +
                        static_cast<std::size_t>(buffer)];
    }

    /// Creates the plan's per-rank command lists if it has none.  Called at
    /// setup so the cost does not land on the first inference, and again from
    /// build_plan for safety.  No-op on the immediate path.
    void ensure_plan_lists(Plan& plan);

    /// True when the collective's events are cleared by commands appended to
    /// the tail of the recorded command lists instead of by zeEventHostReset
    /// in execute_plan.  The host reset is one driver round-trip per event on
    /// every collective (64 per model step); the device-side reset is one
    /// command-processor slot on a list that is already being drained.  The
    /// saving grows with the world size: N=2 has 2 events, the N>2 funnel has
    /// 2*(N-1)+1.  Excluded are the immediate path, which records nothing, and
    /// profiling, which reads kernel timestamps back from those same events
    /// after the sync and would find them wiped.  TP_DEVICE_EVENT_RESET=0
    /// forces the host reset back on so the two can be compared in one build.
    bool use_device_event_reset() const {
        return m_device_event_reset && !m_use_immediate && !m_profiling_enabled;
    }

    bool ensure_scratch_capacity(std::size_t payload_bytes);
    void destroy_scratch();
    void* scratch_buffer(int index, int buffer) const;

    // Per-call host-side breakdown of execute_plan().  Shaped so it stays
    // meaningful for any world size: the per-rank sync times are folded into
    // "the first queue we waited on" and "everything after it" instead of a
    // per-rank vector that would allocate on a path taken 64 times per step.
    struct ExecStats {
        std::chrono::nanoseconds reset{};       // time spent resetting events on the host
        std::chrono::nanoseconds submit{};      // all ExecuteCommandLists calls
        std::chrono::nanoseconds sync_first{};  // first zeCommandQueueSynchronize
        std::chrono::nanoseconds sync_rest{};   // sum of the remaining ones
        // Device-side durations queried via kernel timestamps after sync.
        // Aggregated along the critical path: concurrent transfers are folded
        // with max, sequential phases are added.
        std::chrono::nanoseconds dev_copy{};    // cross-device memcpy
        std::chrono::nanoseconds dev_kernel{};  // reduce kernel(s)
        std::chrono::nanoseconds dev_ts_query{};// time spent in zeEventQueryKernelTimestamp
        // Payload of a single cross-device transfer (n*elem_bytes).  Used for
        // per-link bandwidth, which is what the hardware limit is expressed in.
        std::size_t              copy_bytes{0};
        // Every byte that crosses a device boundary during the collective.
        // N=2 moves 2*payload (one transfer each way); the N>2 funnel moves
        // 2*(N-1)*payload, all of it through rank 0's links.
        std::size_t              copy_bytes_total{0};
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

    /// The body behind allreduce() and allreduce_async().  A null
    /// `model_queue` means the caller wants the collective drained before it
    /// returns; a handle means splice it and go.
    void run_allreduce(int collective_id,
                       int rank,
                       void* in_dev,
                       void* out_dev,
                       std::size_t n,
                       ov::element::Type dtype,
                       ze_command_list_handle_t model_queue);

    /// Watches the per-rank progress counters and turns a device-side wait
    /// that stopped advancing into an aborted group.  Started on the first
    /// note_collective_started(), so a purely synchronous run never pays for
    /// the thread.
    void start_watchdog();
    void stop_watchdog();
    void watchdog_loop();

    /// Signals every event of every plan from the host.  Whatever a device is
    /// waiting on is released, which lets the queues drain and the failure
    /// surface as an exception instead of a hang.  The results of any
    /// collective in flight are garbage, which is why this is only ever
    /// called after the group has already been declared dead.
    void release_all_waits();

    TPL0SharedContextPtr            m_shared;
    int                             m_world_size{0};
    int                             m_num_collectives{0};
    bool                            m_ready{false};
    std::chrono::milliseconds       m_collective_timeout{5000};
    std::atomic<bool>               m_aborted{false};
    mutable std::mutex              m_abort_mutex;
    std::string                     m_abort_reason;

    // Watchdog accounting.  Monotonic, written only by the rank it belongs
    // to: how many collectives this rank has handed to the model's queue.
    // The watchdog uses it as the sign that the host is still moving; whether
    // the devices are is answered by polling the completion events.
    std::unique_ptr<std::atomic<uint64_t>[]> m_started;
    std::once_flag                  m_watchdog_once;
    std::thread                     m_watchdog;
    std::mutex                      m_watchdog_mutex;
    std::condition_variable         m_watchdog_cv;
    bool                            m_watchdog_stop{false};
    // Toggle between regular-cmdlist + queue-sync (false) and
    // immediate-cmdlist + counter-based event sync (true).  Driven by
    // env var TP_USE_IMMEDIATE.  Latched at construction time.
    bool                            m_use_immediate{false};

    // Latched TP_PROF state.  Kernel-timestamp events are only created when
    // this is set, and their values must survive until execute_plan reads
    // them back, which forbids the device-side event reset.
    bool                            m_profiling_enabled{false};

    // Latched TP_DEVICE_EVENT_RESET.  On by default; exists so the host and
    // device reset schemes can be compared without a rebuild.
    bool                            m_device_event_reset{true};

    // Ring instead of the rank-0 funnel for N>2.  Latched at construction
    // because it decides the scratch layout.  TP_RING=0 restores the funnel.
    bool                            m_use_ring{false};

    // Whether the ring's command lists are created with
    // ZE_COMMAND_LIST_FLAG_IN_ORDER.  In-order execution would express the
    // ring's linear chain for free, but combining it with the explicit
    // cross-device events the ring needs makes queues intermittently fail to
    // drain -- reproduced as a 5 s timeout on one rank's queue while the
    // barrier variant passed in the same session.  Default is therefore the
    // explicit barriers; TP_RING_IN_ORDER=1 re-enables the flag for
    // experiments once the driver behaviour is understood.
    bool                            m_ring_in_order{false};

    std::vector<RankState>          m_ranks;        // [N]

    mutable std::mutex              m_scratch_mutex;
    ScratchArena                    m_scratch;

    // One rendezvous + one cached plan per collective_id.
    std::vector<std::unique_ptr<Rendezvous>> m_rendezvous;
    // One recording per slot, deliberately.  Generation alternates between the
    // prompt-sized prefill and the single-token decode, so every switch
    // re-records every collective, and keeping both shapes recorded would
    // obviously avoid that.  It is not safe: the only thing identifying a
    // buffer here is its address, a recorded command list holds the driver's
    // allocation objects in its residency list, and intel_gpu frees and
    // reallocates network buffers between inferences.  A new allocation
    // landing on an old address makes matches() report a hit and the next
    // submit walks a destroyed GraphicsAllocation -- a segfault inside
    // zeCommandQueueExecuteCommandLists, reproducible on the second benchmark
    // iteration.  Re-recording is what clears the stale residency, so it
    // cannot simply be skipped.  The cost disappears on its own once the
    // collective moves into the GPU plugin's own stream and stops being
    // recorded ahead of time at all.
    std::vector<std::unique_ptr<Plan>>       m_plans;

    // Where recording time goes, split by stage, in nanoseconds.  Atomic
    // because each rank records its own lists and all of them accumulate here.
    std::atomic<uint64_t>                    m_rec_drain_ns{0};
    std::atomic<uint64_t>                    m_rec_reset_ns{0};
    std::atomic<uint64_t>                    m_rec_build_ns{0};
    std::atomic<uint64_t>                    m_rec_close_ns{0};

    /// zeCommandListClose with its cost attributed to m_rec_close_ns: closing
    /// is where the driver finalizes the list, and it is the stage most likely
    /// to dominate re-recording.
    void close_list(ze_command_list_handle_t list);
};

using TPDeviceCoordinatorPtr = std::shared_ptr<TPDeviceCoordinator>;

}  // namespace tp_gpu
}  // namespace ov
