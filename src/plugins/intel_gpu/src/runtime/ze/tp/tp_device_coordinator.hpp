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
#include "ze/ze_common.hpp"
#include "collective_options.hpp"
#include "tp_debug.hpp"

namespace cldnn {
namespace ze {
namespace tp {

struct TPL0SharedContext;
using TPL0SharedContextPtr = std::shared_ptr<TPL0SharedContext>;

/// \brief Owns per-rank Level Zero resources and runs cross-GPU collectives
/// directly on device memory through a shared L0 context.
///
/// One per TP group in a process: the group state every rank's communicator
/// shares (see RankCommunicator).
///
/// Lifetime: holds the `TPL0SharedContext` it was created on, so the context
/// outlives it. All per-rank queues and command lists are scoped to the
/// shared context. It builds no kernels: the reduction kernel belongs to the
/// compiled model and comes with every allreduce() call.
///
/// Three schedules, chosen per recording: two ranks exchange directly, larger
/// worlds use a ring, and a power-of-two world carrying a small payload uses
/// recursive halving. Every rank records and submits its own commands -- no
/// rank drives the others.
///
/// A rank's recording is kept until its own buffers, its reduction kernel,
/// the element count or dtype, or a peer's published staging change. Every
/// call goes through the call rendezvous, keyed by the per-rank call number.
class TPDeviceCoordinator {
public:
    /// \param shared       Shared L0 context (driver + N devices + ze_context).
    /// \param world_size   Number of ranks.
    /// \param num_collectives  Number of distinct collective points in the model.
    /// \param collective_timeout  Upper bound on any single wait inside a
    ///        collective, host- or device-side. Zero means wait forever.
    TPDeviceCoordinator(TPL0SharedContextPtr shared,
                        int world_size,
                        int num_collectives,
                        std::chrono::milliseconds collective_timeout = std::chrono::milliseconds{5000},
                        const CollectiveOptions& options = CollectiveOptions{});

    TPDeviceCoordinator(const TPDeviceCoordinator&) = delete;
    TPDeviceCoordinator& operator=(const TPDeviceCoordinator&) = delete;

    ~TPDeviceCoordinator();

    // ---------------------------------------------------------------
    // Topology and state
    // ---------------------------------------------------------------

    /// Returns world size (number of ranks).
    int world_size() const { return m_world_size; }

    /// The device-group context the coordinator runs on.
    const TPL0SharedContextPtr& shared_context() const { return m_shared; }

    /// Number of distinct collective points the coordinator was sized for.
    /// Recorded in the compiled blob so an imported model can rebuild an
    /// identically sized coordinator without re-analyzing the graph.
    int num_collectives() const { return m_num_collectives; }

    /// Returns true once the per-rank queues and plans exist.
    bool is_ready() const { return m_ready; }

    /// True once a collective has failed. The coordinator is single-use after
    /// that: the device queues may still hold unfinished work, so every later
    /// call throws instead of running on top of unknown state.
    bool is_aborted() const { return m_aborted.load(std::memory_order_acquire); }

    // ---------------------------------------------------------------
    // Liveness and failure
    // ---------------------------------------------------------------

    /// Tells the watchdog that this rank has handed a collective to a queue
    /// whose completion the host will not wait for.
    ///
    /// Draining our own queue used to be what caught a dead peer: a rank that
    /// never signalled left `zeCommandQueueSynchronize` to time out. Once the
    /// collective rides in the model's queue there is no such call -- the wait
    /// happens on the device, where Level Zero offers no deadline at all, and
    /// a rank that dies takes the whole inference down into a hang.
    void note_collective_started(int rank);

    /// Whether the driver has the splice extension at all.
    bool async_supported() const;

    /// Whether the collectives should run spliced into the model's queue.
    /// False either because the driver has no splice extension or because a
    /// debug option asked for the synchronous path.
    bool run_spliced() const {
        return async_supported() && !m_options.force_sync_collective;
    }

    const CollectiveOptions& options() const { return m_options; }

    /// Marks the group as failed, wakes every waiting rank and makes all
    /// subsequent calls throw. Safe to call from any rank; the first reason
    /// wins. Must not be called while holding a rendezvous mutex.
    void abort_all(const std::string& reason);

    // ---------------------------------------------------------------
    // Diagnostics
    // ---------------------------------------------------------------

    /// Coordinator-owned device-USM scratch accounting. These allocations
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

    // ---------------------------------------------------------------
    // Collectives
    // ---------------------------------------------------------------

    /// In-place all-reduce-sum across all ranks.
    ///
    /// All N ranks must call this method concurrently (one thread per rank)
    /// for the same `collective_id` with consistent `n`/`dtype`. On return
    /// each rank's `out_dev` holds the elementwise sum across all `in_dev`s.
    ///
    /// `in_dev` and `out_dev` may alias. Both must be device USM allocated
    /// in the shared L0 context on the rank's device.
    ///
    /// `dtype` must be f16, bf16 or f32.
    ///
    /// `reduce_kernel` is the rank's reduction kernel for `dtype`, created in
    /// the shared context on the rank's device; see append_reduce() for
    /// what it computes. It is baked into the recording and must stay alive
    /// while a recording that uses it may still run.
    ///
    /// With `model_queue` -- the immediate command list intel_gpu already
    /// runs the model on -- the recording is spliced there and the call
    /// returns without waiting. The queue is in-order, so the recording
    /// lands after the operations that produced `in_dev` and before whatever
    /// reads `out_dev`, with no drain in between: the host goes on
    /// dispatching the next stretch of the model while the devices work
    /// through the collective. Without a queue the collective runs on the
    /// coordinator's own queues and the call blocks until it is done.
    ///
    /// A queue is honoured only when run_spliced() agrees, so a driver
    /// without the extension and force_sync_collective both land on the
    /// blocking path no matter what the caller passed.
    void allreduce(int collective_id,
                   int rank,
                   void* in_dev,
                   void* out_dev,
                   std::size_t n,
                   ov::element::Type dtype,
                   ze_kernel_handle_t reduce_kernel,
                   ze_command_list_handle_t model_queue = nullptr);

    /// Collects each rank's slice of a row-major matrix into rank 0's buffer.
    ///
    /// Every rank holds `rows` x `slice_elems` elements of its own; rank 0's
    /// buffer is `rows` x (`slice_elems` * world_size), and rank r's slice
    /// belongs at column offset r * slice_elems of every row -- so the slices
    /// are strided in the destination, not contiguous.
    ///
    /// Only rank 0's buffer is written. That is what the vocabulary
    /// projection needs: the infer request reads outputs from rank 0 alone,
    /// so gathering there instead of to everyone saves world_size-1 transfers
    /// per rank for a result nobody would look at.
    ///
    /// `out_dev` is the destination on rank 0 and ignored on every other rank.
    ///
    /// `model_queue` works exactly as it does for allreduce(). It matters
    /// more here: blocking drains the rank's model stream and waits for a copy
    /// on the coordinator's own queue, once per token on the vocabulary
    /// projection. Splicing removes both waits, and with them what made the
    /// blocking form correct -- the host barrier no longer implies the slices
    /// have landed, because the ranks now write from independent queues. The
    /// recording carries that ordering instead: every other rank signals when
    /// its slice is down and the root's recording waits on all of them, so the
    /// model queue of the rank that reads the gathered buffer is held until it
    /// is whole.
    void gather_to_root(int collective_id,
                        int rank,
                        void* in_dev,
                        void* out_dev,
                        std::size_t rows,
                        std::size_t slice_elems,
                        ov::element::Type dtype,
                        ze_command_list_handle_t model_queue = nullptr);

private:
    // ---------------------------------------------------------------
    // Types
    // ---------------------------------------------------------------

    // Per-rank L0 state.
    struct RankState {
        ze_device_handle_t          device{nullptr};
        uint32_t                    compute_ordinal{0};
        ze_command_queue_handle_t   compute_queue{nullptr};

        // Filled at init: GPU timer period (ns/tick) and the mask used to
        // wrap raw kernel-timestamp counter values.
        uint64_t                    timer_ns_per_tick{0};
        uint64_t                    timestamp_mask{~uint64_t{0}};
    };

    // What one rank's recording was built from. Everything a rank's commands
    // depend on is its own buffers and kernel, the call's shape, and where the
    // peers it writes to have their staging -- never a peer's own buffers --
    // so a rank decides on its own whether its recording is still current.
    struct RankRecording {
        bool                        valid{false};
        void*                       in{nullptr};
        void*                       out{nullptr};
        uint64_t                    in_id{0};
        uint64_t                    out_id{0};
        ze_kernel_handle_t          kernel{nullptr};
        std::size_t                 n{0};
        ov::element::Type           dtype{ov::element::dynamic};
        // Allreduce: every rank's staging half for this call, from the mailbox.
        std::vector<void*>          staging;     // [N]
        std::vector<uint64_t>       staging_ids; // [N]
        // Gather: rows x slice of the local matrix, and the root's destination.
        std::size_t                 rows{0};
        std::size_t                 slice_elems{0};
        void*                       root_out{nullptr};
        uint64_t                    root_out_id{0};

        bool operator==(const RankRecording& o) const {
            return valid == o.valid && in == o.in && out == o.out && in_id == o.in_id && out_id == o.out_id &&
                   kernel == o.kernel && n == o.n && dtype == o.dtype && staging == o.staging &&
                   staging_ids == o.staging_ids && rows == o.rows && slice_elems == o.slice_elems &&
                   root_out == o.root_out && root_out_id == o.root_out_id;
        }
        bool operator!=(const RankRecording& o) const { return !(*this == o); }
    };

    // The recordings of one collective for one call parity.
    struct Plan {
        // What the recorded lists do. A collective id belongs to exactly one
        // of these for the life of the model.
        enum class Kind { allreduce, gather };
        Kind                        kind{Kind::allreduce};

        // Call parity this plan serves; selects the staging half and the
        // event set baked into the recording.
        int                         buffer{0};

        // [r] is written and read only by rank r.
        std::vector<RankRecording>  rec;            // [N]

        // Signalled when this rank's spliced recording has finished on the
        // device. Host-visible, unlike everything else here, because two
        // parties ask about it from the host: the rank itself, before
        // splicing the same list again, and the watchdog, which has no other
        // way to tell a device that is merely behind from one that is stuck.
        ze_event_pool_handle_t      done_pool{nullptr};
        std::vector<ze_event_handle_t> ev_done;     // [N]
        // Whether that event belongs to a splice that has not been accounted
        // for yet. Atomic because the watchdog reads it while the ranks
        // write it.
        std::unique_ptr<std::atomic<uint8_t>[]> in_flight;  // [N]

        // What the splice that set in_flight[r] is moving across a device
        // boundary. Held per rank until the completion event of that same
        // splice is read, so the bytes and the duration that get divided into
        // a bandwidth belong to one instance. Counting them at splice time
        // instead let the two advance at different rates, and the first dump
        // divided a prefill's bytes by one decode sample -- reporting a
        // bandwidth several times what the hardware can do.
        std::vector<std::size_t>    spliced_bytes;  // [N]

        // Command lists holding this collective's recorded commands, one per
        // rank. Every collective owns its own so that a recording survives
        // the next collective: with a single list per rank each call
        // overwrote the previous recording, and one decode step re-recorded
        // all 64 collectives for nothing.
        std::vector<ze_command_list_handle_t> compute_lists;  // [N]
    };

    // Events the allreduces of one call parity signal and wait on. Shared by
    // every allreduce of that parity: call k + 2 reuses them only after every
    // rank finished call k + 1, which needs every rank's input of k + 1, which
    // each rank sends only after its call k -- resets included -- completed.
    struct EventSet {
        ze_event_pool_handle_t         pool{nullptr};
        // Pair schedule: [r] = "rank r finished pushing its payload".
        std::vector<ze_event_handle_t> recv;   // [2]
        // Ring and halving: [step * N + rank], signalled by `rank` at `step`
        // and waited on by its partner at that step.
        std::vector<ze_event_handle_t> step;
    };

    // Gather ordering: rank r != 0 signals [r] when its slice is down, and the
    // root waits on all of them. One set per gather parity.
    struct GatherEvents {
        ze_event_pool_handle_t         pool{nullptr};
        std::vector<ze_event_handle_t> ev;     // [N], index 0 unused
    };

    // A rank's staging: two halves, one per allreduce parity. Owned by that
    // rank alone and published per call, so it grows without the group.
    struct Staging {
        void*                       base{nullptr};
        std::size_t                 bytes{0};
        std::size_t                 payload_capacity{0};
        uint64_t                    id{0};
        uint64_t                    growth_count{0};
    };

    // One rank's entry of the call rendezvous.
    struct MailboxSlot {
        uint64_t                    seq{0};
        Plan::Kind                  kind{Plan::Kind::allreduce};
        int                         collective_id{-1};
        std::size_t                 n{0};
        ov::element::Type           dtype{ov::element::dynamic};
        std::size_t                 rows{0};
        // Allreduce: this rank's staging half for the call.
        void*                       staging{nullptr};
        uint64_t                    staging_id{0};
        // Gather, root only: where the slices go.
        void*                       out{nullptr};
        uint64_t                    out_id{0};
    };

    // The call rendezvous. Every rank publishes one slot per call, in the set
    // picked by the parity of the call number, and waits until every rank has
    // published the same call. A set is rewritten only at call k + 2, which a
    // rank reaches after the rendezvous of k + 1 -- so after every peer has
    // published k + 1, which a peer does only after it read the slots of k.
    //
    // A mutex and a condition_variable, deliberately. The rendezvous is
    // crossed once per collective, which looks like an obvious case for a
    // lock-free counter -- but an atomic barrier that spins before parking
    // measured worse at four ranks: what a rank waits for here is mostly its
    // peers being genuinely late, and a spinning waiter takes the core that
    // the late peer's dispatch thread needs.
    struct Mailbox {
        std::mutex                  mtx;
        std::condition_variable     cv;
        std::vector<MailboxSlot>    slots[2];       // [2][N]
        int                         arrived[2]{0, 0};
        uint64_t                    released[2]{0, 0};
        // When the first rank reached the call, used to measure how far
        // behind the others are. Only written when profiling is on.
        std::chrono::steady_clock::time_point first_arrival[2]{};
    };

    // A counter a std::vector can hold. The slots are written by their own
    // rank and read by rank 0 at dump time, so they have to be atomic; plain
    // std::atomic is neither copyable nor movable and a vector of them cannot
    // be sized with assign(), hence the wrapper. Relaxed throughout: these
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

    // ---------------------------------------------------------------
    // Setup and teardown
    // ---------------------------------------------------------------

    void init_rank(RankState& rs);
    void destroy_rank(RankState& rs);

    void destroy_plan(Plan& plan);

    /// Creates the plan's completion events and command lists. How many there
    /// are depends only on the world size, never on the payload, so this runs
    /// once at construction. Doing it lazily put 64 pool creations on the
    /// first inference, which is the one whose latency users measure as time
    /// to first token.
    void create_plan_events(Plan& plan);

    /// Creates the plan's per-rank command lists if it has none.
    void ensure_plan_lists(Plan& plan);

    /// The allreduce and gather event sets of both parities.
    void create_event_sets();
    void destroy_event_sets();

    // ---------------------------------------------------------------
    // Recording
    // ---------------------------------------------------------------

    /// Drains, resets and re-records one rank's lists.
    void record_rank(Plan& plan, int rank);

    /// Records the direct two-rank exchange for one rank: push our input into
    /// the peer's staging, then reduce our input with what the peer pushed
    /// into ours.
    void record_pair_rank(Plan& plan, int rank);

    /// Records the ring schedule for one rank: N-1 reduce-scatter steps
    /// followed by N-1 all-gather steps, sending only to its successor. Each
    /// link carries 2*(N-1)/N of the payload, so the cost per link does not
    /// grow with N.
    void record_ring_rank(Plan& plan, int rank);

    /// Records the recursive halving/doubling exchange for one rank.
    ///
    /// Only for world sizes that are a power of two. Where the ring needs
    /// 2*(N-1) equal-sized steps, this needs 2*log2(N) steps of halving size:
    /// S/2, S/4 ... and back. The link traffic is identical -- 1.5*S per
    /// rank, the optimum -- so what it buys is fewer commands and fewer
    /// round trips, which is what a decode collective is actually made of.
    ///
    /// Every step still talks to exactly one partner (r XOR (1<<d)), which is
    /// the property that matters: schemes needing all ranks to meet at once
    /// pay the arrival skew twice over and lose to the ring despite issuing
    /// fewer commands.
    void record_halving_rank(Plan& plan, int rank);

    /// Records one rank's contribution to a gather: a single strided copy of
    /// its slice into the root's buffer. Ranks write disjoint columns, so
    /// there is no data to order between them, but the root still has to know
    /// when the columns are down: every other rank signals ev_gather[r] and
    /// the root's recording waits on all of them before it ends. That wait is
    /// the only thing holding the root's queue back on either path.
    void record_gather_rank(Plan& plan, int rank);

    /// zeCommandListClose with its cost attributed to `into`: closing is where
    /// the driver finalizes the list, and it is the stage most likely to
    /// dominate re-recording. The accumulator is a parameter so the gather,
    /// which does not share the rest of the recording breakdown, does not
    /// contaminate it.
    void close_list(ze_command_list_handle_t list, Counter& into);

    // ---------------------------------------------------------------
    // Staging and its addressing
    // ---------------------------------------------------------------

    /// Makes `rank`'s staging large enough for a payload of this size and
    /// returns its half for call parity `buffer`. Growing waits for the rank's
    /// own previous collective to finish on its device: by then every peer
    /// write into the old staging has landed, because that collective waited
    /// for them.
    void* ensure_staging(int rank, std::size_t payload_bytes, int buffer, int collective_id);
    void destroy_staging();

    /// Bytes of one staging half for a payload of this size.
    std::size_t half_bytes(std::size_t payload_bytes) const;

    /// Ring: distance between chunk slots inside a staging half.
    std::size_t chunk_stride(std::size_t payload_bytes) const;

    /// Where the peer's payload lands on `rank`. Two ranks only.
    void* pair_stage(const RankRecording& rec, int rank) const;

    /// Element range of ring chunk `chunk` within a payload of `n` elements.
    void ring_chunk(std::size_t n, int chunk, std::size_t& offset, std::size_t& count) const;

    /// Staging slot for `chunk` on `rank`.
    void* ring_slot(const RankRecording& rec, int rank, int chunk) const;

    /// The element range this rank owns after `level` halvings of the payload.
    void halving_range(std::size_t n, int rank, int level,
                       std::size_t& lo, std::size_t& hi) const;

    /// Where a partner deposits data on `rank` for one halving step. Regions
    /// [0, levels) take the scatter steps, [levels, 2 * levels) the gather
    /// steps; every step has its own, because a partner may deliver the next
    /// step while this rank still reads the last.
    void* halving_stage(const RankRecording& rec, int rank, int region) const;

    /// Whether a payload of this size is reduced by recursive halving.
    bool uses_halving(std::size_t payload_bytes) const;

    /// Staging bytes per rank and parity that a payload of this size needs.
    std::size_t staging_bytes(std::size_t payload_bytes) const;

    /// Two sets of plans, staging halves and events, by call parity.
    int plan_buffers() const { return 2; }

    /// The plan holding `collective_id` for the given parity.
    Plan& plan_at(int collective_id, int buffer) const {
        return *m_plans[static_cast<std::size_t>(collective_id) *
                            static_cast<std::size_t>(plan_buffers()) +
                        static_cast<std::size_t>(buffer)];
    }

    // ---------------------------------------------------------------
    // Call rendezvous
    // ---------------------------------------------------------------

    /// Publishes `mine` for call `mine.seq` and waits until every rank has
    /// published the same call; `all` receives every rank's slot. Ranks that
    /// disagree on which collective this is, or on its shape, abort the group.
    void call_rendezvous(int rank, const MailboxSlot& mine, std::vector<MailboxSlot>& all);

    // ---------------------------------------------------------------
    // Handing work to the devices
    // ---------------------------------------------------------------

    /// Submits one rank's lists. Every rank must be submitted before any of
    /// them is waited on: a rank's list blocks on events its neighbours only
    /// signal once they run, so draining one first would deadlock.
    void submit_rank(Plan& plan, int rank);

    /// Waits for one rank's queues to drain.
    void sync_rank(Plan& plan, int rank);

    /// Waits until the previous splice of this rank's recording has finished,
    /// so the list can be appended or re-recorded. Returns whether there was one.
    bool await_previous_splice(Plan& plan, int rank, int collective_id);

    // ---------------------------------------------------------------
    // Timeouts and failure handling
    // ---------------------------------------------------------------

    /// Timeout translated to the nanosecond argument L0 sync calls take.
    /// Zero timeout maps to "no limit".
    uint64_t timeout_ns() const;

    /// zeCommandQueueSynchronize bounded by the collective timeout. Throws on
    /// expiry so a stuck queue surfaces as an error instead of a frozen process.
    void sync_queue(ze_command_queue_handle_t queue, const char* what) const;

    /// Throws if a previous collective already failed.
    void throw_if_aborted() const;

    /// Aborts the group and throws. `timed_out` distinguishes "this rank ran
    /// out of patience" from "somebody else already failed".
    [[noreturn]] void fail_collective(bool timed_out, int collective_id, int rank, const char* stage);

    /// Watches the per-rank progress counters and turns a device-side wait
    /// that stopped advancing into an aborted group. Started on the first
    /// note_collective_started(), so a purely synchronous run never pays for
    /// the thread.
    void start_watchdog();
    void stop_watchdog();
    void watchdog_loop();

    /// Signals every event of every plan from the host. Whatever a device is
    /// waiting on is released, which lets the queues drain and the failure
    /// surface as an exception instead of a hang. The results of any
    /// collective in flight are garbage, which is why this is only ever
    /// called after the group has already been declared dead.
    void release_all_waits();

    // ---------------------------------------------------------------
    // Profiling
    //
    // Measurement only. With GPU_TP_PROFILING unset nothing below is written,
    // and in a build without GPU debug caps the option does not exist and
    // profiling stays off.
    // ---------------------------------------------------------------

    /// Buckets of the device-occupancy histogram. Bucket b covers
    /// [2^b, 2^(b+1)) microseconds, so 20 of them reach a full second -- past
    /// anything a collective should ever take, and the top one is a catch-all
    /// anyway.
    static constexpr int kDevBuckets = 20;

    // How far apart the ranks arrive at a collective, and what each of them
    // spends its time waiting for. Every counter is indexed by rank and
    // written only by that rank, except the totals, which are written under
    // the rendezvous mutex by whichever rank arrives last.
    struct SkewStats {
        CounterVec ph1_ns;      // [N] time spent in the call rendezvous
        CounterVec ph2_ns;      // [N] time spent in the execute phase
        CounterVec late_ns;     // [N] arrival minus the first arrival
        CounterVec last_count;  // [N] how often this rank arrived last
        // The stretch of model work between leaving one collective and
        // reaching the next. This is where the arrival skew is built, so it
        // is measured per rank with its extremes kept, not just averaged.
        CounterVec seg_ns;      // [N] sum of segment durations
        CounterVec seg_min_ns;  // [N]
        CounterVec seg_max_ns;  // [N]
        CounterVec seg_count;   // [N]
        std::vector<std::chrono::steady_clock::time_point> last_exit;  // [N]
        // Breakdown of ph2 on the splice path. The phase looked far more
        // expensive than the append itself costs, and the pieces are not
        // guessable from the outside: the gate is a condvar handoff for every
        // rank but rank 0, and the event wait is only supposed to fire when
        // the host has run two collectives ahead of the device.
        CounterVec p2_gate_ns;    // [N] staging check (and growth) before the rendezvous
        CounterVec p2_rec_ns;     // [N] recording this rank's commands
        CounterVec p2_wait_ns;    // [N] zeEventHostSynchronize on the previous splice
        CounterVec p2_reset_ns;   // [N] zeEventHostReset
        CounterVec p2_append_ns;  // [N] the splice itself
        CounterVec p2_rec_count;  // [N] how often a re-record happened
        CounterVec p2_wait_count; // [N] how often the previous splice was still in flight
        CounterVec p2_block_count;// [N] how often that wait actually had to block
        // How long the spliced collective occupied the rank's queue, read off
        // the completion event's kernel timestamps. This is the one part of
        // the cost that no host phase contains: the splice hands the work over
        // and returns, so the copies, the reduce kernel and the waits on peers
        // all happen after every host measurement has ended. Sampled where
        // the previous splice is already being queried anyway, so it costs one
        // extra call per collective and no synchronization.
        CounterVec p2_dev_ns;     // [N] sum of (kernelEnd - kernelStart)
        CounterVec p2_dev_max_ns; // [N]
        CounterVec p2_dev_count;  // [N]
        // Which collective was responsible for that maximum. Every slot is
        // averaged together otherwise, and they are not the same size: knowing
        // whether one of them dominates decides whether there is a single
        // thing to optimize or a uniform cost to live with.
        CounterVec p2_dev_max_cid; // [N]
        // The mean above is useless on its own -- prefill and decode differ by
        // two orders of magnitude and land in the same average. A histogram
        // over powers of two separates them without keeping samples.
        // Flat [N * kDevBuckets]; bucket b holds durations in [2^b, 2^(b+1)) us.
        CounterVec p2_dev_hist;
        // Device time this rank spent NOT in a collective, between the end of
        // one and the start of the next. Both numbers come off completion
        // events that are read anyway, so this costs nothing and answers the
        // question the whole design rests on: what share of the token does the
        // collective actually own.
        CounterVec dev_gap_ns;    // [N]
        CounterVec dev_gap_count; // [N]
        // Where the previous collective ended on this rank, in that device's
        // raw ticks, and which collective it was. Written and read only by
        // the owning rank, so they need no atomics; the id is what keeps the
        // wrap from the last collective back to the first -- a gap containing
        // the whole rest of the model -- out of the average.
        std::vector<uint64_t> dev_last_end_ticks;  // [N]
        std::vector<int>      dev_last_cid;        // [N]
        // Bytes this rank pushed across a device boundary, summed over the
        // collectives it took part in. Arithmetic rather than measurement:
        // the schedule fixes exactly how much every rank sends, so counting it
        // is exact and free, and dividing it by the queue occupancy above is
        // the only bandwidth figure the production path can produce.
        CounterVec dev_bytes;     // [N]
        // Which schedule the recordings used. Written by the recording rank.
        CounterVec n_pair;        // [N]
        CounterVec n_ring;        // [N]
        CounterVec n_halving;     // [N]
        uint64_t spread_ns{0};             // sum of (last arrival - first arrival)
    };

    /// The gather kept apart from the allreduces. It is one collective among
    /// many but nothing like the others -- one strided copy per rank into the
    /// root, sized by the vocabulary rather than the hidden dimension -- so
    /// folding it into the same averages would hide both it and them.
    struct GatherStats {
        CounterVec barrier_ns;  // [N] the call rendezvous
        CounterVec append_ns;   // [N] the splice itself
        CounterVec calls;       // [N]
        CounterVec spliced;     // [N] how many of those rode the model queue
        CounterVec records;     // [N] how often the recording had to be redone
        CounterVec dev_ns;      // [N] queue occupancy, from the completion event
        CounterVec dev_max_ns;  // [N]
        CounterVec dev_count;   // [N]
        CounterVec bytes;       // [N] what this rank copies into the root
    };

    /// How far the host runs ahead of the devices, which is the entire point
    /// of splicing and was until now the one thing nobody measured. Kept as a
    /// running count of this rank's handed-over-but-unaccounted splices: up on
    /// every splice, down when its completion event is consumed.
    struct RunAhead {
        CounterVec now;    // [N] outstanding right now
        CounterVec sum;    // [N] sum of `now` sampled at every splice
        CounterVec max;    // [N]
        CounterVec count;  // [N]
    };

    /// Rank-0 totals for the whole coordinator. These used to be function
    /// statics inside allreduce(), which made them process-global: two
    /// models in one process added their numbers together and neither report
    /// meant anything. Only rank 0 writes them and only rank 0 reads them at
    /// dump time, but they are Counters anyway so the type matches the rest.
    struct HostTotals {
        Counter record_ns;   // re-recording this rank's lists
        Counter splice_ns;   // handing the recording over
        Counter prep_ns;     // rendezvous exit to the start of plan work
        Counter tail_ns;     // handover done to returning
        Counter rebuilds;    // staging growths
        Counter records;     // re-recordings
    };

    /// Closes out one collective for the measurement bookkeeping: advances the
    /// call counter and emits the reports when a dump period has gone by.
    ///
    /// Every collective must call this exactly once, from rank 0 and any rank
    /// for the counter to stay honest -- allreduce, gather, and whatever is
    /// added next. Keeping the decision here rather than in one operation's
    /// body is what stops a new collective from silently not counting: the
    /// gather used to advance the counter but never close a period, so its
    /// share of the work never produced a report.
    void note_collective_done(int rank);

    /// Prints the accumulated measurements. Split out of note_collective_done
    /// so the decision to report and the reporting itself stay separable.
    void emit_report();

    /// The three sections of a report. `calls` is how many collectives rank 0
    /// has taken part in; every per-call figure divides by it.
    void report_host_ranks(uint64_t calls);
    void report_host_totals(uint64_t calls);
    void report_device(uint64_t calls);

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------

    TPL0SharedContextPtr            m_shared;
    int                             m_world_size{0};
    int                             m_num_collectives{0};
    bool                            m_ready{false};
    std::chrono::milliseconds       m_collective_timeout{5000};
    std::atomic<bool>               m_aborted{false};
    mutable std::mutex              m_abort_mutex;
    std::string                     m_abort_reason;

    // Watchdog accounting. Monotonic, written only by the rank it belongs
    // to: how many collectives this rank has handed to the model's queue.
    // The watchdog uses it as the sign that the host is still moving; whether
    // the devices are is answered by polling the completion events.
    std::unique_ptr<std::atomic<uint64_t>[]> m_started;
    std::once_flag                  m_watchdog_once;
    std::thread                     m_watchdog;
    std::mutex                      m_watchdog_mutex;
    std::condition_variable         m_watchdog_cv;
    bool                            m_watchdog_stop{false};

    // Resolved from the GPU_TP_* options when the TP context was created.
    CollectiveOptions               m_options;

    // Ring reduce-scatter + all-gather for N>2; two ranks exchange directly,
    // which is what the ring degenerates to minus a round of latency.
    // Derived from the world size alone and latched at construction because
    // it decides the scratch layout.
    bool                            m_use_ring{false};

    std::vector<RankState>          m_ranks;        // [N]

    // Per rank, touched only by that rank (stats read them under the lock).
    mutable std::mutex              m_staging_mutex;
    std::vector<Staging>            m_staging;      // [N]
    std::vector<uint64_t>           m_seq;          // [N] calls of any kind
    std::vector<uint64_t>           m_allreduce_seq;// [N] allreduce calls: their parity
    std::vector<uint64_t>           m_gather_seq;   // [N] gather calls: their parity
    // The plan this rank spliced last, whose completion means everything it
    // handed over before is done too.
    std::vector<Plan*>              m_last_splice;  // [N]

    Mailbox                         m_mailbox;
    EventSet                        m_events[2];
    GatherEvents                    m_gather_events[2];
    // One recording per slot, deliberately. Generation alternates between the
    // prompt-sized prefill and the single-token decode, so every switch
    // re-records every collective, and keeping both shapes recorded would
    // obviously avoid that. It is not safe: a recorded command list holds the
    // driver's allocation objects in its residency list, and intel_gpu frees
    // and reallocates network buffers between inferences. A new allocation
    // landing on an old address used to make matches() report a hit and the
    // next submit walked a destroyed GraphicsAllocation -- a segfault inside
    // zeCommandQueueExecuteCommandLists. The signature now carries the
    // driver's allocation id as well as the address, so that case is caught
    // and re-records; what re-recording is for is clearing the stale
    // residency, so it cannot simply be skipped. The cost disappears on its
    // own once the collective moves into the GPU plugin's own stream and stops
    // being recorded ahead of time at all.
    std::vector<std::unique_ptr<Plan>>       m_plans;

    // ---------------------------------------------------------------
    // Profiling state
    // ---------------------------------------------------------------

    SkewStats                       m_skew;
    GatherStats                     m_gather;
    RunAhead                        m_ahead;
    HostTotals                      m_totals;

    // Counts every collective rank 0 takes part in and says when a report is
    // due. A period that has passed, not a count that divides: rank 0 makes
    // one allreduce call per allreduce slot, while the period defaults to the
    // number of slots, which also counts the gather. The two differ by one,
    // so an exact-division test would land on a multiple only by coincidence.
    DumpPeriod                      m_dump_period;

    // Where recording time goes, split by stage, in nanoseconds. Shared by
    // every rank: each records its own lists and all of them accumulate here.
    Counter                         m_rec_drain_ns;
    Counter                         m_rec_reset_ns;
    Counter                         m_rec_build_ns;
    Counter                         m_rec_close_ns;
    /// Closing a gather recording, kept apart: the gather never goes through
    /// record_rank, so folding its close into m_rec_close_ns made the
    /// "append" figure -- build minus close -- able to come out negative.
    Counter                         m_gather_close_ns;

    /// Time ranks spend growing their staging, which waits for their own
    /// device to finish the previous collective. Rare, but it lands in time to
    /// first token.
    Counter                         m_scratch_stall_ns;
};

using TPDeviceCoordinatorPtr = std::shared_ptr<TPDeviceCoordinator>;

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
