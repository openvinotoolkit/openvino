// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "collective_options.hpp"
#include "openvino/core/type/element_type.hpp"
#include "tp_debug.hpp"
#include "ze/ze_common.hpp"

namespace cldnn {
namespace ze {
namespace tp {

struct TPL0SharedContext;
using TPL0SharedContextPtr = std::shared_ptr<TPL0SharedContext>;

/// The Level Zero entry points, resolved once. The ov::ze* wrappers take a
/// process-global lock on every call, which serializes every Level Zero call
/// made anywhere in OpenVINO; the collectives call through this table instead.
const std::shared_ptr<ov::ZeroApi>& ze_api();

/// The driver's id of the allocation holding `ptr`, or the largest id when the
/// driver no longer knows the pointer. Part of every recording signature: an
/// address alone does not tell a freed buffer from a new one at the same place.
uint64_t alloc_id_of(ze_context_handle_t context, void* ptr);

/// One hand-over of a recorded command list to a device queue. Owned by the
/// recording; the group polls it (watchdog), signals it (abort) and waits for
/// it (staging growth), so the recording registers it with track() and
/// withdraws it with untrack() before destroying it.
struct Splice {
    ze_event_handle_t done{nullptr};
    /// Set while `done` belongs to a hand-over that has not been accounted for.
    std::atomic<uint8_t> in_flight{0};
    /// What the hand-over moves across a device boundary; device profiling only.
    std::size_t bytes{0};
    bool allreduce{true};
};

/// A counter a std::vector can hold. Relaxed throughout: these are statistics,
/// and no other memory is published through them.
struct Counter {
    std::atomic<uint64_t> v{0};
    Counter() = default;
    explicit Counter(uint64_t init) : v(init) {}
    Counter(const Counter& o) : v(o.get()) {}
    Counter& operator=(const Counter& o) {
        v.store(o.get(), std::memory_order_relaxed);
        return *this;
    }
    uint64_t get() const {
        return v.load(std::memory_order_relaxed);
    }
    void add(uint64_t d) {
        v.fetch_add(d, std::memory_order_relaxed);
    }
    void bump() {
        add(1);
    }
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

/// What GPU_TP_PROFILING measures. Written by the recordings and the group,
/// printed by the group. Every vector is indexed by rank and written only by
/// that rank. With profiling off nothing here is written.
struct CollectiveStats {
    /// Buckets of the device-occupancy histogram: bucket b covers [2^b, 2^(b+1)) us.
    static constexpr int kDevBuckets = 20;

    // Arrival skew and the host phases of an allreduce.
    CounterVec ph1_ns;      // staging check + call rendezvous
    CounterVec ph2_ns;      // leaving the rendezvous to having handed the work over
    CounterVec late_ns;     // arrival minus the first arrival
    CounterVec last_count;  // how often this rank arrived last
    // Model work between leaving one collective and reaching the next.
    CounterVec seg_ns;
    CounterVec seg_min_ns;
    CounterVec seg_max_ns;
    CounterVec seg_count;
    std::vector<std::chrono::steady_clock::time_point> last_exit;
    CounterVec p2_gate_ns;     // staging check (and growth) before the rendezvous
    CounterVec p2_rec_ns;      // recording this rank's commands
    CounterVec p2_wait_ns;     // waiting for the previous splice of the list
    CounterVec p2_reset_ns;    // zeEventHostReset
    CounterVec p2_append_ns;   // the splice itself
    CounterVec p2_rec_count;
    CounterVec p2_wait_count;
    CounterVec p2_block_count;
    // Queue occupancy of a spliced allreduce, from its completion event.
    CounterVec p2_dev_ns;
    CounterVec p2_dev_max_ns;
    CounterVec p2_dev_count;
    CounterVec p2_dev_max_cid;
    CounterVec p2_dev_hist;  // flat [N * kDevBuckets]
    // Device time between the end of one collective and the start of the next.
    CounterVec dev_gap_ns;
    CounterVec dev_gap_count;
    std::vector<uint64_t> dev_last_end_ticks;
    std::vector<int> dev_last_cid;
    CounterVec dev_bytes;
    // Which schedule the recordings used.
    CounterVec n_pair;
    CounterVec n_ring;
    CounterVec n_halving;
    uint64_t spread_ns{0};  // sum of (last arrival - first arrival), under the mailbox lock

    // The gather, kept out of the allreduce averages.
    CounterVec gather_barrier_ns;
    CounterVec gather_append_ns;
    CounterVec gather_calls;
    CounterVec gather_spliced;
    CounterVec gather_records;
    CounterVec gather_dev_ns;
    CounterVec gather_dev_max_ns;
    CounterVec gather_dev_count;
    CounterVec gather_bytes;

    // Splices handed over that the devices have not caught up with.
    CounterVec ahead_now;
    CounterVec ahead_sum;
    CounterVec ahead_max;
    CounterVec ahead_count;

    // Rank-0 totals.
    Counter record_ns;
    Counter splice_ns;
    Counter prep_ns;
    Counter tail_ns;
    Counter rebuilds;
    Counter records;

    // Recording time by stage, summed over all ranks.
    Counter rec_drain_ns;
    Counter rec_reset_ns;
    Counter rec_build_ns;
    Counter rec_close_ns;
    Counter gather_close_ns;
    Counter scratch_stall_ns;

    explicit CollectiveStats(int world_size);
};

/// \brief The collective resources of one TP group in a process, shared by
/// every rank's communicator.
///
/// Resources only: the call rendezvous, every rank's staging, the events the
/// ranks signal each other, a queue per rank for the synchronous fallback,
/// failure handling and the watchdog. What runs over them -- the schedules and
/// their command lists -- belongs to the collective implementations, which
/// reach this through their channel.
class GroupState {
public:
    /// \param collective_timeout  Upper bound on any single wait, host- or
    ///        device-side. Zero means wait forever.
    GroupState(TPL0SharedContextPtr shared,
               int world_size,
               int num_collectives,
               std::chrono::milliseconds collective_timeout = std::chrono::milliseconds{5000},
               const CollectiveOptions& options = CollectiveOptions{});
    GroupState(const GroupState&) = delete;
    GroupState& operator=(const GroupState&) = delete;
    ~GroupState();

    int world_size() const {
        return m_world_size;
    }
    const TPL0SharedContextPtr& shared_context() const {
        return m_shared;
    }
    ze_context_handle_t context() const;
    int num_collectives() const {
        return m_num_collectives;
    }
    bool is_ready() const {
        return m_ready;
    }
    const CollectiveOptions& options() const {
        return m_options;
    }
    ze_device_handle_t device(int rank) const {
        return m_ranks[static_cast<std::size_t>(rank)].device;
    }
    uint32_t compute_ordinal(int rank) const {
        return m_ranks[static_cast<std::size_t>(rank)].compute_ordinal;
    }

    /// Whether the driver has the splice extension at all.
    bool async_supported() const;
    /// Whether collectives run spliced into the model's queue: the driver can,
    /// and no debug option asked for the synchronous path.
    bool run_spliced() const {
        return async_supported() && !m_options.force_sync_collective;
    }

    // ---------------------------------------------------------------
    // Calls
    // ---------------------------------------------------------------

    /// What an allreduce learns from the call rendezvous.
    struct AllreduceCall {
        int parity{0};
        std::vector<void*> staging;        // [N] every rank's staging half for this call
        std::vector<uint64_t> staging_ids; // [N]
        std::vector<ze_event_handle_t>* recv{nullptr};  // pair: [r] = "rank r pushed its payload"
        std::vector<ze_event_handle_t>* step{nullptr};  // ring/halving: [step * N + rank]
    };

    /// Makes `rank`'s staging hold two halves of `half_bytes`, then meets the
    /// other ranks at the call rendezvous. Throws, and aborts the group, if the
    /// ranks disagree on the call or one of them does not arrive in time.
    void begin_allreduce(int rank,
                         int collective_id,
                         std::size_t n,
                         ov::element::Type dtype,
                         std::size_t half_bytes,
                         AllreduceCall& call);

    /// What a gather learns from the call rendezvous.
    struct GatherCall {
        int parity{0};
        void* root_staging{nullptr};  // the root's half for this call
        uint64_t root_staging_id{0};
        std::vector<ze_event_handle_t>* events{nullptr};  // [r] = "rank r's slice is down", r > 0
        std::vector<ze_event_handle_t>* acks{nullptr};    // [r] = "the root has taken rank r's slice", r > 0
    };

    /// The call rendezvous of a gather. On the root, first makes its gather
    /// staging hold two halves of `half_bytes`.
    void begin_gather(int rank,
                      int collective_id,
                      std::size_t rows,
                      std::size_t slice_elems,
                      ov::element::Type dtype,
                      std::size_t half_bytes,
                      GatherCall& call);

    // ---------------------------------------------------------------
    // Hand-over
    // ---------------------------------------------------------------

    /// A recording's command lists and completion events, one per call parity.
    struct RecordingResources {
        ze_event_pool_handle_t pool{nullptr};
        ze_command_list_handle_t lists[2]{};
        ze_event_handle_t done[2]{};
    };

    /// The resources made for (`collective_id`, `rank`) when the group was
    /// created, or new ones once those are taken. The caller owns them
    /// afterwards and releases them with destroy_recording_resources().
    /// Made ahead because creating them costs about a millisecond per
    /// collective and rank, which would otherwise land on the first inference.
    RecordingResources take_recording_resources(int collective_id, int rank);
    static void destroy_recording_resources(RecordingResources& res);

    /// Registers a recording's splice with the watchdog and the abort path.
    void track(Splice* splice);
    /// Withdraws it. The caller has made sure it is not in flight.
    void untrack(int rank, Splice* splice);

    /// Waits until `splice` has finished on the device and clears it. Returns
    /// whether it was in flight.
    bool await_splice(Splice& splice, int rank, int collective_id);

    /// Records that `rank` has just handed `splice` to its model queue.
    void handed_over(int rank, Splice* splice);

    /// The synchronous fallback: runs `list` on the rank's own queue and waits.
    void execute_and_wait(int rank, ze_command_list_handle_t list, const char* what);
    /// Waits for the rank's own queue to drain.
    void drain(int rank, const char* what);

    // ---------------------------------------------------------------
    // Failure
    // ---------------------------------------------------------------

    /// True once a collective has failed. The group is single-use after that.
    bool is_aborted() const {
        return m_aborted.load(std::memory_order_acquire);
    }
    /// Marks the group as failed, wakes every waiting rank and makes all later
    /// calls throw. The first reason wins.
    void abort_all(const std::string& reason);
    /// Throws if a previous collective already failed.
    void throw_if_aborted() const;
    /// Aborts the group and throws. `timed_out` distinguishes "this rank ran
    /// out of patience" from "somebody else already failed".
    [[noreturn]] void fail_collective(bool timed_out, int collective_id, int rank, const char* stage);
    /// The timeout in the nanoseconds Level Zero waits take; zero maps to no limit.
    uint64_t timeout_ns() const;

    /// Tells the watchdog that `rank` has handed a collective to a queue whose
    /// completion the host will not wait for.
    void note_collective_started(int rank);

    // ---------------------------------------------------------------
    // Diagnostics
    // ---------------------------------------------------------------

    /// Group-owned device-USM staging, which bypasses the intel_gpu memory pool.
    struct ScratchStats {
        std::size_t payload_capacity_bytes{0};
        std::size_t total_allocated_bytes{0};
        std::vector<std::size_t> allocated_bytes_per_rank;
        uint64_t generation{0};
        uint64_t growth_count{0};
        uint64_t allocation_count{0};
    };
    ScratchStats get_scratch_stats() const;

    CollectiveStats& stats() {
        return m_stats;
    }
    uint64_t timer_ns_per_tick(int rank) const {
        return m_ranks[static_cast<std::size_t>(rank)].timer_ns_per_tick;
    }
    uint64_t timestamp_mask(int rank) const {
        return m_ranks[static_cast<std::size_t>(rank)].timestamp_mask;
    }

    /// Closes out one collective for the measurement bookkeeping and prints a
    /// report when a dump period has gone by. Every collective calls it once.
    void note_collective_done(int rank);

private:
    struct RankState {
        ze_device_handle_t device{nullptr};
        uint32_t compute_ordinal{0};
        ze_command_queue_handle_t compute_queue{nullptr};
        uint64_t timer_ns_per_tick{0};
        uint64_t timestamp_mask{~uint64_t{0}};
    };

    // Events the allreduces of one call parity signal and wait on. Call k + 2
    // reuses them only after every rank finished call k + 1, which needs every
    // rank's input of k + 1, which each rank sends only after its call k --
    // resets included -- completed.
    struct EventSet {
        ze_event_pool_handle_t pool{nullptr};
        std::vector<ze_event_handle_t> recv;  // pair: [2]
        std::vector<ze_event_handle_t> step;  // ring/halving: [(2(N-1)+1) * N]
    };

    // Gather ordering, one set per gather parity. Rank r != 0 signals ev[r]
    // when its slice is in the root's staging; the root signals ack[r] once it
    // has copied that slice out, and rank r waits for it before writing the
    // same half again. Peers depend on nothing else of the root, so without
    // the ack two gathers in a row could overwrite a half still being read.
    struct GatherEvents {
        ze_event_pool_handle_t pool{nullptr};
        std::vector<ze_event_handle_t> ev;   // [N], index 0 unused
        std::vector<ze_event_handle_t> ack;  // [N], index 0 unused; created signalled
    };

    // A rank's staging: two halves, one per allreduce parity, each half of the
    // allocation. Owned by that rank alone and published per call.
    struct Staging {
        void* base{nullptr};
        std::size_t bytes{0};
        std::size_t payload_capacity{0};
        uint64_t id{0};
        uint64_t growth_count{0};
    };

    // One rank's entry of the call rendezvous.
    struct MailboxSlot {
        uint64_t seq{0};
        bool allreduce{true};
        int collective_id{-1};
        std::size_t n{0};
        ov::element::Type dtype{ov::element::dynamic};
        std::size_t rows{0};
        void* staging{nullptr};  // allreduce: this rank's staging half; gather: the root's
        uint64_t staging_id{0};
    };

    // The call rendezvous. Every rank publishes one slot per call, in the set
    // picked by the parity of the call number, and waits until every rank has
    // published the same call. A set is rewritten only at call k + 2, which a
    // rank reaches after the rendezvous of k + 1 -- so after every peer has
    // published k + 1, which a peer does only after it read the slots of k.
    //
    // A mutex and a condition_variable, deliberately: an atomic barrier that
    // spins before parking measured worse at four ranks, because a spinning
    // waiter takes the core the late peer's dispatch thread needs.
    struct Mailbox {
        std::mutex mtx;
        std::condition_variable cv;
        std::vector<MailboxSlot> slots[2];
        int arrived[2]{0, 0};
        uint64_t released[2]{0, 0};
        std::chrono::steady_clock::time_point first_arrival[2]{};
    };

    void init_rank(RankState& rs);
    void destroy_rank(RankState& rs);
    void create_event_sets();
    void destroy_event_sets();
    RecordingResources create_recording_resources(int rank) const;

    /// Makes `s` -- `rank`'s staging -- hold two halves of `half_bytes` and
    /// returns the half for `parity`. Growing waits for the rank's own last
    /// hand-over: by then every peer write into the old staging has landed,
    /// because that collective waited for them.
    void* ensure_staging(Staging& s,
                         int rank,
                         std::size_t payload_bytes,
                         std::size_t half_bytes,
                         int parity,
                         int collective_id);
    void destroy_staging();

    /// Publishes `mine` and waits until every rank has published the same call;
    /// `all` receives every rank's slot.
    void call_rendezvous(int rank, const MailboxSlot& mine, std::vector<MailboxSlot>& all);

    void sync_queue(ze_command_queue_handle_t queue, const char* what) const;

    void start_watchdog();
    void stop_watchdog();
    void watchdog_loop();
    /// Signals every event from the host so whatever a device waits on is
    /// released. Only after the group has been declared dead.
    void release_all_waits();

    void emit_report();
    void report_host_ranks(uint64_t calls);
    void report_host_totals(uint64_t calls);
    void report_device(uint64_t calls);

    TPL0SharedContextPtr m_shared;
    int m_world_size{0};
    int m_num_collectives{0};
    bool m_ready{false};
    std::chrono::milliseconds m_collective_timeout{5000};
    CollectiveOptions m_options;

    std::atomic<bool> m_aborted{false};
    mutable std::mutex m_abort_mutex;
    std::string m_abort_reason;

    // How many collectives each rank has handed to the model's queue: the
    // watchdog's sign that the hosts are still moving.
    std::unique_ptr<std::atomic<uint64_t>[]> m_started;
    std::once_flag m_watchdog_once;
    std::thread m_watchdog;
    std::mutex m_watchdog_mutex;
    std::condition_variable m_watchdog_cv;
    bool m_watchdog_stop{false};

    // Every live recording's splices; the watchdog and the abort path walk them.
    std::mutex m_tracked_mutex;
    std::vector<Splice*> m_tracked;

    std::vector<RankState> m_ranks;

    mutable std::mutex m_staging_mutex;
    std::vector<Staging> m_staging;
    // The root's landing area for gathers, apart from the allreduce staging:
    // the two count their parities separately.
    Staging m_gather_staging;
    std::vector<uint64_t> m_seq;            // [N] calls of any kind: the mailbox set
    std::vector<uint64_t> m_allreduce_seq;  // [N] their parity picks staging half and events
    std::vector<uint64_t> m_gather_seq;     // [N] kept apart: a gather waits on nobody
    // The splice each rank handed over last; its completion means everything
    // the rank handed over before is done too.
    std::vector<Splice*> m_last_splice;

    Mailbox m_mailbox;
    EventSet m_events[2];
    GatherEvents m_gather_events[2];

    std::mutex m_made_ahead_mutex;
    std::vector<RecordingResources> m_made_ahead;  // [collective * N + rank]

    CollectiveStats m_stats;
    DumpPeriod m_dump_period;
};

using GroupStatePtr = std::shared_ptr<GroupState>;

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
