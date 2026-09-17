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
#include "tp_gpu/tp_config.hpp"

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
                        std::chrono::milliseconds collective_timeout = std::chrono::milliseconds{5000},
                        const TPConfig& config = TPConfig{});

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

    /// Whether the driver has the splice extension at all.
    virtual bool async_supported() const;

    /// Whether the collectives should run spliced into the model's queue.
    /// False either because the driver has no splice extension or because a
    /// debug option asked for the synchronous path.  Read by the primitives
    /// living in the intel_gpu plugin, which have no config object of their
    /// own -- and reading it here is also what keeps a `getenv` off the hot
    /// path, where it used to run on every collective of every inference.
    bool run_spliced() const {
        return async_supported() && !m_config.force_sync_collective();
    }

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
    ///
    /// With `model_queue` -- the immediate command list intel_gpu already
    /// runs the model on -- the recording is spliced there and the call
    /// returns without waiting.  The queue is in-order, so the recording
    /// lands after the operations that produced `in_dev` and before whatever
    /// reads `out_dev`, with no drain in between: the host goes on
    /// dispatching the next stretch of the model while the devices work
    /// through the collective.  Without a queue the collective runs on the
    /// coordinator's own queues and the call blocks until it is done.
    ///
    /// A queue is honoured only when run_spliced() agrees, so a driver
    /// without the extension and force_sync_collective both land on the
    /// blocking path no matter what the caller passed.
    ///
    /// Virtual because the GPU plugin calls the coordinator through this
    /// vtable without linking against the plugin that defines it.
    virtual void allreduce(int collective_id,
                           int rank,
                           void* in_dev,
                           void* out_dev,
                           std::size_t n,
                           ov::element::Type dtype,
                           ze_command_list_handle_t model_queue = nullptr);

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
    ///
    /// `model_queue` works exactly as it does for allreduce().  It matters
    /// more here: blocking drains the rank's model stream and waits for a copy
    /// on the coordinator's own queue, once per token on the vocabulary
    /// projection.  Splicing removes both waits, and with them what made the
    /// blocking form correct -- the host barrier no longer implies the slices
    /// have landed, because the ranks now write from independent queues.  The
    /// recording carries that ordering instead: every other rank signals when
    /// its slice is down and the root's recording waits on all of them, so the
    /// model queue of the rank that reads the gathered buffer is held until it
    /// is whole.
    virtual void gather_to_root(int collective_id,
                                int rank,
                                void* in_dev,
                                void* out_dev,
                                std::size_t rows,
                                std::size_t slice_elems,
                                ov::element::Type dtype,
                                ze_command_list_handle_t model_queue = nullptr);

private:
    // Per-rank L0 state.
    struct RankState {
        ze_device_handle_t          device{nullptr};
        uint32_t                    compute_ordinal{0};
        ze_command_queue_handle_t   compute_queue{nullptr};
        ze_command_list_handle_t    compute_list{nullptr};

        ze_module_handle_t          module{nullptr};
        ze_kernel_handle_t          kernel_f16{nullptr};
        ze_kernel_handle_t          kernel_f32{nullptr};

        // Filled at init: GPU timer period (ns/tick) and the mask used to
        // wrap raw kernel-timestamp counter values.
        uint64_t                    timer_ns_per_tick{0};
        uint64_t                    timestamp_mask{~uint64_t{0}};
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

        // Which schedule the recorded lists implement.  Chosen per recording
        // by record_rank from the world size and the payload, and kept here
        // because the events a rank is allowed to clear depend on it: every
        // event has exactly one waiter, and which ones those are differs
        // between the three schedules.
        enum class Schedule { pair, ring, halving };
        Schedule                    schedule{Schedule::pair};

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
        // Pair schedule only: [r] = "rank r finished pushing its payload".
        std::vector<ze_event_handle_t> ev_recv;     // [2]

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

        // What the splice that set in_flight[r] is moving across a device
        // boundary.  Held per rank until the completion event of that same
        // splice is read, so the bytes and the duration that get divided into
        // a bandwidth belong to one instance.  Counting them at splice time
        // instead made the two run at different rates -- the first dump
        // divided a prefill's bytes by one decode sample and reported
        // 734 GB/s.
        std::vector<std::size_t>    spliced_bytes;  // [N]

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

        // Optional device-side timestamp probes (N=2 path): ev_ts_kernel[r]
        // is signaled by rank r's reduce kernel.  Allocated out of `pool`,
        // which carries KERNEL_TIMESTAMP, and only when device profiling is
        // on -- entries stay null otherwise and act as a "no signal"
        // sentinel for AppendLaunchKernel.  The memcpy end is probed through
        // ev_recv[r], which the copy signals anyway.
        std::vector<ze_event_handle_t> ev_ts_kernel;  // [N]

        // Command lists holding this collective's recorded commands, one per
        // rank.  Every collective owns its own so that a recording survives
        // the next collective: with a single list per rank each call
        // overwrote the previous recording, and one decode step re-recorded
        // all 64 collectives for nothing.  Empty on the immediate path,
        // which appends directly at execute time.
        std::vector<ze_command_list_handle_t> compute_lists;  // [N]

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
    //     payload, the same order as TP=2.
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
        // Two sets, picked by the parity of the entry generation.  Without the
        // exit barrier a rank that has already left can reach this collective
        // again and publish its pointers while a slower peer is still reading
        // the ones from the instance it is in; alternating the sets keeps the
        // two apart.  Reaching the set again means passing the enter barrier
        // twice, which the slow peer itself has to take part in, so a set is
        // never rewritten while anyone still needs it.
        std::vector<void*>          in_ptrs[2];     // [2][N], filled by ranks
        std::vector<void*>          out_ptrs[2];    // [2][N], filled by ranks
        int                         arrived{0};
        int                         departed{0};
        bool                        done{false};
        // Bumped by rank 0 once a collective's recording decision is settled.
        // A counter rather than the `done` flag because nothing clears it any
        // more: the exit barrier that used to reset the flag is gone.
        uint64_t                    record_gen{0};
        uint64_t                    enter_gen{0};
        uint64_t                    exit_gen{0};
        std::size_t                 n{0};
        ov::element::Type           dtype{ov::element::dynamic};
        // When the first rank reached this collective, used to measure how
        // far behind the others are.  Only written when profiling is on.
        std::chrono::steady_clock::time_point first_arrival{};
    };

    // A counter a std::vector can hold.  The slots are written by their own
    // rank and read by rank 0 at dump time, so they have to be atomic; plain
    // std::atomic is neither copyable nor movable and a vector of them cannot
    // be sized with assign(), hence the wrapper.  Relaxed throughout: these
    // are statistics, and no other memory is published through them.
    struct Counter {
        std::atomic<uint64_t> v{0};
        Counter() = default;
        explicit Counter(uint64_t init) : v(init) {}
        Counter(const Counter& o) : v(o.get()) {}
        Counter& operator=(const Counter& o) {
            v.store(o.get(), std::memory_order_relaxed);
            return *this;
        }
        uint64_t get() const { return v.load(std::memory_order_relaxed); }
        void add(uint64_t d) { v.fetch_add(d, std::memory_order_relaxed); }
        void bump() { add(1); }
        void keep_max(uint64_t d) {
            if (d > get()) {
                v.store(d, std::memory_order_relaxed);
            }
        }
        void keep_min(uint64_t d) {
            if (d < get()) {
                v.store(d, std::memory_order_relaxed);
            }
        }
    };
    using CounterVec = std::vector<Counter>;

    /// Buckets of the device-occupancy histogram.  Bucket b covers
    /// [2^b, 2^(b+1)) microseconds, so 20 of them reach a full second -- past
    /// anything a collective should ever take, and the top one is a catch-all
    /// anyway.
    static constexpr int kDevBuckets = 20;

    // How far apart the ranks arrive at a collective, and what each of them
    // spends its time waiting for.  Collected only when TP_PROFILING is set to
    // anything but NONE.  Every counter is indexed by rank and written only by
    // that rank, except the totals, which are written under the rendezvous
    // mutex by whichever rank arrives last.
    struct SkewStats {
        CounterVec ph1_ns;      // [N] time spent in the enter barrier
        CounterVec ph2_ns;      // [N] time spent in the execute phase
        CounterVec late_ns;     // [N] arrival minus the first arrival
        CounterVec last_count;  // [N] how often this rank arrived last
        // The stretch of model work between leaving one collective and
        // reaching the next.  This is where the arrival skew is built, so it
        // is measured per rank with its extremes kept, not just averaged.
        CounterVec seg_ns;      // [N] sum of segment durations
        CounterVec seg_min_ns;  // [N]
        CounterVec seg_max_ns;  // [N]
        CounterVec seg_count;   // [N]
        std::vector<std::chrono::steady_clock::time_point> last_exit;  // [N]
        // Breakdown of ph2 on the splice path.  The phase looked far more
        // expensive than the 1.3 us the append itself costs, and the pieces
        // are not guessable from the outside: the gate is a condvar handoff
        // for every rank but rank 0, and the event wait is only supposed to
        // fire when the host has run two collectives ahead of the device.
        CounterVec p2_gate_ns;    // [N] rank 0: prep; others: wait for the record gate
        CounterVec p2_rec_ns;     // [N] recording this rank's commands
        CounterVec p2_wait_ns;    // [N] zeEventHostSynchronize on the previous splice
        CounterVec p2_reset_ns;   // [N] zeEventHostReset
        CounterVec p2_append_ns;  // [N] the splice itself
        CounterVec p2_rec_count;  // [N] how often a re-record happened
        CounterVec p2_wait_count; // [N] how often the previous splice was still in flight
        CounterVec p2_block_count;// [N] how often that wait actually had to block
        // How long the spliced collective occupied the rank's queue, read off
        // the completion event's kernel timestamps.  This is the one part of
        // the cost that no host phase contains: the splice hands the work over
        // and returns, so the copies, the reduce kernel and the waits on peers
        // all happen after every host measurement has ended.  Sampled where
        // the previous splice is already being queried anyway, so it costs one
        // extra call per collective and no synchronization.
        CounterVec p2_dev_ns;     // [N] sum of (kernelEnd - kernelStart)
        CounterVec p2_dev_max_ns; // [N]
        CounterVec p2_dev_count;  // [N]
        // Which collective was responsible for that maximum.  All 65 slots are
        // averaged together otherwise, and they are not the same size: knowing
        // whether one of them dominates decides whether there is a single
        // thing to optimize or a uniform cost to live with.
        CounterVec p2_dev_max_cid; // [N]
        // The mean above is useless on its own -- prefill and decode differ by
        // two orders of magnitude and land in the same average.  A histogram
        // over powers of two separates them without keeping samples.
        // Flat [N * kDevBuckets]; bucket b holds durations in [2^b, 2^(b+1)) us.
        CounterVec p2_dev_hist;
        // Device time this rank spent NOT in a collective, between the end of
        // one and the start of the next.  Both numbers come off completion
        // events that are read anyway, so this costs nothing and answers the
        // question the whole design rests on: what share of the token does the
        // collective actually own.
        CounterVec dev_gap_ns;    // [N]
        CounterVec dev_gap_count; // [N]
        // Where the previous collective ended on this rank, in that device's
        // raw ticks, and which collective it was.  Written and read only by
        // the owning rank, so they need no atomics; the id is what keeps the
        // wrap from collective 64 back to 0 -- a gap containing the whole rest
        // of the model -- out of the average.
        std::vector<uint64_t> dev_last_end_ticks;  // [N]
        std::vector<int>      dev_last_cid;        // [N]
        // Bytes this rank pushed across a device boundary, summed over the
        // collectives it took part in.  Arithmetic rather than measurement:
        // the schedule fixes exactly how much every rank sends, so counting it
        // is exact and free, and dividing it by the queue occupancy above is
        // the only bandwidth figure the production path can produce.
        CounterVec dev_bytes;     // [N]
        // Which schedule the recordings used.  Written by the recording rank.
        CounterVec n_pair;        // [N]
        CounterVec n_ring;        // [N]
        CounterVec n_halving;     // [N]
        // The record gate has a fast path a rank takes when it can see for
        // itself that neither the signature nor the arena moved.  It is the
        // difference between a condvar handoff on every collective and almost
        // none, so how often it actually holds is worth knowing rather than
        // assuming.
        CounterVec gate_fast;     // [N]
        CounterVec gate_slow;     // [N]
        uint64_t spread_ns{0};             // sum of (last arrival - first arrival)
        uint64_t calls{0};
        // Value of `calls` at the previous dump.  The trigger is "a period has
        // passed", not "the count divides by the period": rank 0 makes one
        // allreduce call per allreduce slot, but the period defaults to the
        // number of collective slots, which also counts the gather.  The two
        // differ by one, so an exact-division test lands on a multiple only by
        // coincidence and a short run prints nothing at all.
        uint64_t last_dump{0};
    };
    SkewStats                       m_skew;

    /// The gather kept apart from the allreduces.  It is one collective out of
    /// sixty-five but nothing like the others -- one strided copy per rank into
    /// the root, sized by the vocabulary rather than the hidden dimension -- so
    /// folding it into the same averages would hide both it and them.
    struct GatherStats {
        CounterVec barrier_ns;  // [N] the enter barrier
        CounterVec append_ns;   // [N] the splice itself
        CounterVec calls;       // [N]
        CounterVec spliced;     // [N] how many of those rode the model queue
        CounterVec records;     // [N] how often the recording had to be redone
        CounterVec dev_ns;      // [N] queue occupancy, from the completion event
        CounterVec dev_max_ns;  // [N]
        CounterVec dev_count;   // [N]
        CounterVec bytes;       // [N] what this rank copies into the root
    };
    GatherStats                     m_gather;

    /// How far the host runs ahead of the devices, which is the entire point
    /// of splicing and was until now the one thing nobody measured.  Kept as a
    /// running count of this rank's handed-over-but-unaccounted splices: up on
    /// every splice, down when its completion event is consumed.
    struct RunAhead {
        CounterVec now;    // [N] outstanding right now
        CounterVec sum;    // [N] sum of `now` sampled at every splice
        CounterVec max;    // [N]
        CounterVec count;  // [N]
    };
    RunAhead                        m_ahead;

    /// Rank-0 totals for the whole coordinator.  These used to be function
    /// statics inside allreduce(), which made them process-global: two
    /// models in one process added their numbers together and neither report
    /// meant anything.  Only rank 0 writes them and only rank 0 reads them at
    /// dump time, but they are Counters anyway so the type matches the rest.
    struct HostTotals {
        Counter ph1_ns;
        Counter ph2_ns;
        Counter record_ns;   // the record gate
        Counter splice_ns;   // handing the recording over
        Counter prep_ns;     // barrier exit to the start of plan work
        Counter tail_ns;     // handover done to returning
        Counter rebuilds;    // arena growths
        Counter records;     // re-recordings
    };
    HostTotals                      m_totals;

    void init_rank(RankState& rs);
    void destroy_rank(RankState& rs);

    void destroy_plan(Plan& plan);

    /// Creates the plan's event pool and its events.  How many there are
    /// depends only on the world size and on whether profiling is on, never on
    /// the payload, so this runs once at construction.  Doing it lazily put 64
    /// pool creations on the first inference, which is the one whose latency
    /// users measure as time to first token.
    void create_plan_events(Plan& plan);

    /// Drains, resets and re-records one rank's lists.
    void record_rank(Plan& plan, int rank);

    /// Submits one rank's lists.  Every rank must be submitted before any of
    /// them is waited on: a rank's list blocks on events its neighbours only
    /// signal once they run, so draining one first would deadlock.
    void submit_rank(Plan& plan, int rank);

    /// Waits for one rank's queues to drain.
    void sync_rank(Plan& plan, int rank);

    /// Records the ring schedule for one rank: N-1 reduce-scatter steps
    /// followed by N-1 all-gather steps, sending only to its successor.  Each
    /// link carries 2*(N-1)/N of the payload, so the cost per link does not
    /// grow with N.
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

    /// Records the recursive halving/doubling exchange for one rank.
    ///
    /// Only for world sizes that are a power of two.  Where the ring needs
    /// 2*(N-1) equal-sized steps, this needs 2*log2(N) steps of halving size:
    /// S/2, S/4 ... and back.  The link traffic is identical -- 1.5*S per
    /// rank, the optimum -- so what it buys is fewer commands and fewer
    /// round trips, which is what a decode collective is actually made of.
    ///
    /// Every step still talks to exactly one partner (r XOR (1<<d)), which is
    /// the property that matters: schemes needing all ranks to meet at once
    /// pay the arrival skew twice over and lose to the ring despite issuing
    /// fewer commands.
    void record_halving_rank(Plan& plan, int rank);

    /// The element range this rank owns after `level` halvings of the payload.
    void halving_range(std::size_t n, int rank, int level,
                       std::size_t& lo, std::size_t& hi) const;

    /// Where the partner of step `step` deposits its half on this rank.  One
    /// region per step, because a partner may deliver step j+1 while this
    /// rank's kernel is still reading what arrived at step j.
    void* halving_stage(int rank, int step, int buffer) const;

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
    /// command list, an event or a byte of the arena.
    int plan_buffers() const { return 2; }

    /// The plan holding `collective_id` in the given buffer.
    Plan& plan_at(int collective_id, int buffer) const {
        return *m_plans[static_cast<std::size_t>(collective_id) *
                            static_cast<std::size_t>(plan_buffers()) +
                        static_cast<std::size_t>(buffer)];
    }

    /// Creates the plan's per-rank command lists if it has none.  Called at
    /// setup so the cost does not land on the first inference, and again from
    /// build_plan for safety.
    void ensure_plan_lists(Plan& plan);

    /// Closes out one collective for the measurement bookkeeping: advances the
    /// call counter and emits the reports when a dump period has gone by.
    ///
    /// Every collective must call this exactly once, from rank 0 and any rank
    /// for the counter to stay honest -- allreduce, gather, and whatever is
    /// added next.  Keeping the decision here rather than in one operation's
    /// body is what stops a new collective from silently not counting: the
    /// gather used to advance the counter but never close a period, so its
    /// share of the work never produced a report.
    void note_collective_done(int rank);

    /// Prints the accumulated measurements.  Split out of note_collective_done
    /// so the decision to report and the reporting itself stay separable.
    void emit_report();

    bool ensure_scratch_capacity(std::size_t payload_bytes);

    /// Whether ensure_scratch_capacity() would have to allocate.  A plain
    /// read, so a rank can decide on its own whether the arena is about to
    /// move under it without taking part in the gate that settles it.
    bool scratch_needs_growth(std::size_t payload_bytes) const {
        return payload_bytes > m_scratch.payload_capacity_bytes;
    }
    void destroy_scratch();
    void* scratch_buffer(int index, int buffer) const;

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

    // Resolved from the plugin configuration at construction time.
    TPConfig                        m_config;

    // Ring reduce-scatter + all-gather for N>2; two ranks exchange directly,
    // which is what the ring degenerates to minus a round of latency.
    // Derived from the world size alone and latched at construction because
    // it decides the scratch layout.
    bool                            m_use_ring{false};

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
    /// Closing a gather recording, kept apart: the gather never goes through
    /// record_rank, so folding its close into m_rec_close_ns made the
    /// "append" figure -- build minus close -- able to come out negative.
    std::atomic<uint64_t>                    m_gather_close_ns{0};

    /// Time rank 0 spends inside ensure_scratch_capacity when it has to grow
    /// the arena.  That path drains every rank's queues before it can move the
    /// staging addresses, so it is a full stop of the whole group -- rare, but
    /// it lands in time to first token, which is where it is least welcome.
    std::atomic<uint64_t>                    m_scratch_stall_ns{0};

    /// zeCommandListClose with its cost attributed to `into`: closing is where
    /// the driver finalizes the list, and it is the stage most likely to
    /// dominate re-recording.  The accumulator is a parameter so the gather,
    /// which does not share the rest of the recording breakdown, does not
    /// contaminate it.
    void close_list(ze_command_list_handle_t list, std::atomic<uint64_t>& into);
};

using TPDeviceCoordinatorPtr = std::shared_ptr<TPDeviceCoordinator>;

}  // namespace tp_gpu
}  // namespace ov
