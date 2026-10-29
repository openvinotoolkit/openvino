// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_state.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

#include "openvino/core/except.hpp"
#include "tp_l0_shared_context.hpp"

// Turns a failed Level Zero call into an exception naming the call.
#define ZE_THROW(expr) OV_ZE_EXPECT(expr)

// Verbosity travels with the options, so the collectives log the same way whoever created the context.
#define TP_GROUP_LOG(level)                \
    if (m_options.verbosity >= (level)) \
    ::cldnn::ze::tp::log_stream()

namespace cldnn {
namespace ze {
namespace tp {

const std::shared_ptr<ov::ZeroApi>& ze_api() {
    static const std::shared_ptr<ov::ZeroApi> api = ov::ZeroApi::get_instance();
    return api;
}

uint64_t alloc_id_of(ze_context_handle_t context, void* ptr) {
    ze_memory_allocation_properties_t props{ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES,
                                            nullptr,
                                            ZE_MEMORY_TYPE_UNKNOWN,
                                            0,
                                            0};
    ze_device_handle_t device = nullptr;
    if (ze_api()->zeMemGetAllocProperties(context, ptr, &props, &device) != ZE_RESULT_SUCCESS) {
        return std::numeric_limits<uint64_t>::max();
    }
    return props.id;
}

namespace {

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* what) {
    OPENVINO_ASSERT(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
                    "[TP][L0] ", what, " byte count overflow: ", lhs, " * ", rhs);
    return lhs * rhs;
}

void select_compute_ordinal(ze_device_handle_t dev, uint32_t& ordinal) {
    uint32_t qg_count = 0;
    ZE_THROW(ov::zeDeviceGetCommandQueueGroupProperties(dev, &qg_count, nullptr));
    std::vector<ze_command_queue_group_properties_t> qgp(qg_count,
                                                         {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
    ZE_THROW(ov::zeDeviceGetCommandQueueGroupProperties(dev, &qg_count, qgp.data()));
    ordinal = 0;
    for (uint32_t g = 0; g < qg_count; ++g) {
        if (qgp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            ordinal = g;
            return;
        }
    }
    OPENVINO_THROW("[TP][L0] no compute-capable queue group found");
}

enum class WaitOutcome { ready, timed_out, aborted };

/// Blocks until `ready()` holds, the group is aborted, or the timeout expires.
/// A zero timeout means "wait forever".
template <class Ready>
WaitOutcome wait_for_condition(std::unique_lock<std::mutex>& lk,
                               std::condition_variable& cv,
                               std::chrono::milliseconds timeout,
                               const std::atomic<bool>& aborted,
                               Ready ready) {
    const auto pred = [&] {
        return ready() || aborted.load(std::memory_order_acquire);
    };
    if (timeout.count() == 0) {
        cv.wait(lk, pred);
    } else if (!cv.wait_for(lk, timeout, pred)) {
        return WaitOutcome::timed_out;
    }
    // A rank that reached its goal may finish even if another rank aborted meanwhile.
    return ready() ? WaitOutcome::ready : WaitOutcome::aborted;
}

uint64_t elapsed_ns(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
}

}  // namespace

CollectiveStats::CollectiveStats(int world_size) {
    const auto n = static_cast<std::size_t>(world_size);
    for (auto* c : {&ph1_ns,           &ph2_ns,           &late_ns,          &last_count,     &seg_ns,
                    &seg_max_ns,       &seg_count,        &p2_gate_ns,       &p2_rec_ns,      &p2_wait_ns,
                    &p2_reset_ns,      &p2_append_ns,     &p2_rec_count,     &p2_wait_count,  &p2_block_count,
                    &p2_dev_ns,        &p2_dev_max_ns,    &p2_dev_count,     &p2_dev_max_cid, &dev_gap_ns,
                    &dev_gap_count,    &dev_bytes,        &n_pair,           &n_ring,         &n_halving,
                    &gather_barrier_ns, &gather_append_ns, &gather_calls,    &gather_spliced, &gather_records,
                    &gather_dev_ns,    &gather_dev_max_ns, &gather_dev_count, &gather_bytes,  &ahead_now,
                    &ahead_sum,        &ahead_max,        &ahead_count}) {
        c->assign(n, Counter{});
    }
    p2_dev_hist.assign(n * static_cast<std::size_t>(kDevBuckets), Counter{});
    dev_last_end_ticks.assign(n, 0);
    dev_last_cid.assign(n, -1);
    // The only running minimum, so it cannot start at zero.
    seg_min_ns.assign(n, Counter{~uint64_t{0}});
    last_exit.assign(n, std::chrono::steady_clock::time_point{});
}

GroupState::GroupState(TPL0SharedContextPtr shared,
                       int world_size,
                       int num_collectives,
                       std::chrono::milliseconds collective_timeout,
                       const CollectiveOptions& options)
    : m_shared(std::move(shared)),
      m_world_size(world_size),
      m_num_collectives(num_collectives),
      m_collective_timeout(collective_timeout),
      m_options(options),
      m_stats(world_size),
      m_dump_period(options.dump_period) {
    OPENVINO_ASSERT(m_shared && m_shared->context && static_cast<int>(m_shared->devices.size()) == world_size,
                    "[TP][L0] the group requires a valid shared L0 context with ",
                    world_size,
                    " devices");
    OPENVINO_ASSERT(world_size >= 2, "[TP][L0] the group requires at least 2 ranks");
    OPENVINO_ASSERT(m_collective_timeout.count() >= 0, "[TP][L0] collective timeout must not be negative");

    const auto n_ranks = static_cast<std::size_t>(world_size);
    m_ranks.resize(n_ranks);
    for (int r = 0; r < world_size; ++r) {
        m_ranks[static_cast<std::size_t>(r)].device = m_shared->devices[static_cast<std::size_t>(r)];
    }
    m_staging.assign(n_ranks, Staging{});
    m_seq.assign(n_ranks, 0);
    m_allreduce_seq.assign(n_ranks, 0);
    m_gather_seq.assign(n_ranks, 0);
    m_last_splice.assign(n_ranks, nullptr);
    m_mailbox.slots[0].assign(n_ranks, MailboxSlot{});
    m_mailbox.slots[1].assign(n_ranks, MailboxSlot{});
    m_started = std::make_unique<std::atomic<uint64_t>[]>(n_ranks);
    for (std::size_t r = 0; r < n_ranks; ++r) {
        m_started[r].store(0, std::memory_order_relaxed);
    }

    try {
        for (auto& rs : m_ranks) {
            init_rank(rs);
        }
        create_event_sets();
        m_made_ahead.reserve(static_cast<std::size_t>(num_collectives) * n_ranks);
        for (int i = 0; i < num_collectives; ++i) {
            for (int r = 0; r < world_size; ++r) {
                m_made_ahead.push_back(create_recording_resources(r));
            }
        }
    } catch (...) {
        for (auto& res : m_made_ahead) {
            destroy_recording_resources(res);
        }
        destroy_event_sets();
        for (auto& rs : m_ranks) {
            destroy_rank(rs);
        }
        throw;
    }

    m_ready = true;
    TP_GROUP_LOG(ov::log::Level::INFO) << "[TP][L0] group ready: " << world_size << " ranks, " << num_collectives
                                       << " collectives" << std::endl;
}

GroupState::~GroupState() {
    // Before anything it watches goes away. Every recording holds the group,
    // so none is left to use the staging or the events below.
    stop_watchdog();
    for (auto& res : m_made_ahead) {
        destroy_recording_resources(res);
    }
    destroy_staging();
    destroy_event_sets();
    for (auto& rs : m_ranks) {
        destroy_rank(rs);
    }
}

ze_context_handle_t GroupState::context() const {
    return m_shared->context;
}

bool GroupState::async_supported() const {
    return ze_api()->zeCommandListImmediateAppendCommandListsExp != nullptr;
}

GroupState::ScratchStats GroupState::get_scratch_stats() const {
    std::lock_guard<std::mutex> lock(m_staging_mutex);
    ScratchStats stats;
    stats.payload_capacity_bytes = std::numeric_limits<std::size_t>::max();
    for (const auto& s : m_staging) {
        stats.payload_capacity_bytes = std::min(stats.payload_capacity_bytes, s.payload_capacity);
        stats.total_allocated_bytes += s.bytes;
        stats.allocated_bytes_per_rank.push_back(s.bytes);
        stats.growth_count = std::max(stats.growth_count, s.growth_count);
        stats.allocation_count += s.growth_count;
    }
    stats.generation = stats.growth_count;
    return stats;
}

void GroupState::init_rank(RankState& rs) {
    select_compute_ordinal(rs.device, rs.compute_ordinal);

    // Kernel-timestamp -> ns conversion. On Intel GPUs timerResolution is ns/tick.
    ze_device_properties_t dp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
    ZE_THROW(ov::zeDeviceGetProperties(rs.device, &dp));
    rs.timer_ns_per_tick = dp.timerResolution > 0 ? dp.timerResolution : 1;
    const uint32_t bits = dp.kernelTimestampValidBits;
    rs.timestamp_mask = (bits == 0 || bits >= 64) ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);

    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
    qd.ordinal = rs.compute_ordinal;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_THROW(ov::zeCommandQueueCreate(m_shared->context, rs.device, &qd, &rs.compute_queue));
}

void GroupState::destroy_rank(RankState& rs) {
    if (rs.compute_queue) {
        ov::zeCommandQueueDestroy(rs.compute_queue);
        rs.compute_queue = nullptr;
    }
}

GroupState::RecordingResources GroupState::create_recording_resources(int rank) const {
    RecordingResources res;
    try {
        // Host-visible: the rank asks before splicing the same list again, the
        // watchdog asks to tell a device that is behind from one that is stuck.
        ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
        epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        if (m_options.profiling_device) {
            epd.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        }
        epd.count = 2;
        std::vector<ze_device_handle_t> devs(m_shared->devices.begin(), m_shared->devices.end());
        ZE_THROW(ov::zeEventPoolCreate(m_shared->context, &epd, static_cast<uint32_t>(devs.size()), devs.data(),
                                       &res.pool));
        for (int p = 0; p < 2; ++p) {
            ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
            ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
            ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
            ed.index = static_cast<uint32_t>(p);
            ZE_THROW(ov::zeEventCreate(res.pool, &ed, &res.done[p]));

            ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
            ld.commandQueueGroupOrdinal = compute_ordinal(rank);
            ZE_THROW(ov::zeCommandListCreate(m_shared->context, device(rank), &ld, &res.lists[p]));
        }
    } catch (...) {
        destroy_recording_resources(res);
        throw;
    }
    return res;
}

GroupState::RecordingResources GroupState::take_recording_resources(int collective_id, int rank) {
    OPENVINO_ASSERT(collective_id >= 0 && collective_id < m_num_collectives && rank >= 0 && rank < m_world_size,
                    "[TP][L0] no recording resources for collective ",
                    collective_id,
                    " on rank ",
                    rank);
    {
        std::lock_guard<std::mutex> lock(m_made_ahead_mutex);
        auto& made = m_made_ahead[static_cast<std::size_t>(collective_id) * static_cast<std::size_t>(m_world_size) +
                                  static_cast<std::size_t>(rank)];
        if (made.pool != nullptr) {
            return std::exchange(made, RecordingResources{});
        }
    }
    return create_recording_resources(rank);
}

void GroupState::destroy_recording_resources(RecordingResources& res) {
    for (int p = 0; p < 2; ++p) {
        if (res.done[p])
            ov::zeEventDestroy(res.done[p]);
        if (res.lists[p])
            ov::zeCommandListDestroy(res.lists[p]);
    }
    if (res.pool)
        ov::zeEventPoolDestroy(res.pool);
    res = RecordingResources{};
}

void GroupState::create_event_sets() {
    auto ctx = m_shared->context;
    const int N = m_world_size;
    std::vector<ze_device_handle_t> devs(m_shared->devices.begin(), m_shared->devices.end());
    auto make = [&](ze_event_pool_handle_t pool, uint32_t index) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
        ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ed.wait = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ed.index = index;
        ze_event_handle_t ev = nullptr;
        ZE_THROW(ov::zeEventCreate(pool, &ed, &ev));
        return ev;
    };

    for (auto& set : m_events) {
        ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
        if (N == 2) {
            // Kept on the pair pool unconditionally: some Intel L0 drivers route
            // cross-device waits differently for timestamp pools, and this is the
            // validated configuration.
            epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
            epd.count = 2;
        } else {
            // One event per (step, rank) over 2*(N-1) ring steps, plus one
            // "reduce-scatter finished" per rank. Halving's 2*log2(N) steps fit.
            epd.flags = m_options.profiling_device ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
            epd.count = static_cast<uint32_t>((2 * (N - 1) + 1) * N);
        }
        ZE_THROW(ov::zeEventPoolCreate(ctx, &epd, static_cast<uint32_t>(devs.size()), devs.data(), &set.pool));
        auto& events = (N == 2) ? set.recv : set.step;
        for (uint32_t i = 0; i < epd.count; ++i) {
            events.push_back(make(set.pool, i));
        }
    }

    for (auto& set : m_gather_events) {
        ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
        epd.count = static_cast<uint32_t>(2 * N);
        ZE_THROW(ov::zeEventPoolCreate(ctx, &epd, static_cast<uint32_t>(devs.size()), devs.data(), &set.pool));
        for (int i = 0; i < N; ++i) {
            set.ev.push_back(make(set.pool, static_cast<uint32_t>(i)));
            set.ready.push_back(make(set.pool, static_cast<uint32_t>(N + i)));
        }
    }
}

void GroupState::destroy_event_sets() {
    auto destroy = [](std::vector<ze_event_handle_t>& events) {
        for (auto e : events) {
            if (e)
                ov::zeEventDestroy(e);
        }
        events.clear();
    };
    for (auto& set : m_events) {
        destroy(set.recv);
        destroy(set.step);
        if (set.pool)
            ov::zeEventPoolDestroy(set.pool);
        set.pool = nullptr;
    }
    for (auto& set : m_gather_events) {
        destroy(set.ev);
        destroy(set.ready);
        if (set.pool)
            ov::zeEventPoolDestroy(set.pool);
        set.pool = nullptr;
    }
}

void* GroupState::ensure_staging(Staging& s,
                                 int rank,
                                 std::size_t payload_bytes,
                                 std::size_t half_bytes,
                                 int parity,
                                 int collective_id) {
    const std::size_t need = checked_multiply(half_bytes, 2, "staging");
    if (need > s.bytes) {
        ScopedTime stall(m_options.profiling_host, m_stats.scratch_stall_ns);
        if (Splice* last = m_last_splice[static_cast<std::size_t>(rank)]) {
            if (await_splice(*last, rank, collective_id) && m_options.profiling_host && last->allreduce) {
                m_stats.ahead_now[rank].v.fetch_sub(1, std::memory_order_relaxed);
            }
        }
        drain(rank, "staging grow: compute queue drain");

        ze_device_mem_alloc_desc_t mad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr};
        void* fresh = nullptr;
        ZE_THROW(ov::zeMemAllocDevice(m_shared->context, &mad, need, 256, device(rank), &fresh));
        {
            std::lock_guard<std::mutex> lock(m_staging_mutex);
            if (s.base) {
                ov::zeMemFree(m_shared->context, s.base);
            }
            s.base = fresh;
            s.bytes = need;
            s.id = alloc_id_of(m_shared->context, fresh);
            ++s.growth_count;
        }
        if (rank == 0 && m_options.profiling_host) {
            m_stats.rebuilds.bump();
        }
        TP_GROUP_LOG(ov::log::Level::INFO) << "[TP][MEM] staging grow rank=" << rank << " payload=" << payload_bytes
                                           << " bytes=" << need << " growths=" << s.growth_count << std::endl;
    }
    if (payload_bytes > s.payload_capacity) {
        std::lock_guard<std::mutex> lock(m_staging_mutex);
        s.payload_capacity = payload_bytes;
    }
    // Halves split the allocation, not the payload: calls of different sizes
    // must not let one parity's half run into the other's.
    return static_cast<uint8_t*>(s.base) + static_cast<std::size_t>(parity) * (s.bytes / 2);
}

void GroupState::destroy_staging() {
    std::lock_guard<std::mutex> lock(m_staging_mutex);
    for (auto& s : m_staging) {
        if (s.base && m_shared && m_shared->context) {
            ov::zeMemFree(m_shared->context, s.base);
        }
        s = Staging{};
    }
}

void GroupState::call_rendezvous(int rank, const MailboxSlot& mine, std::vector<MailboxSlot>& all) {
    using clk = std::chrono::steady_clock;
    const bool skew = m_options.profiling_host && mine.allreduce;
    const int p = static_cast<int>(mine.seq & 1u);
    auto& mb = m_mailbox;
    auto& st = m_stats;
    {
        std::unique_lock<std::mutex> lk(mb.mtx);
        mb.slots[p][static_cast<std::size_t>(rank)] = mine;
        if (skew) {
            const auto t_arrive = clk::now();
            if (mb.arrived[p] == 0) {
                mb.first_arrival[p] = t_arrive;
            }
            st.late_ns[rank].add(elapsed_ns(mb.first_arrival[p], t_arrive));
            if (mb.arrived[p] + 1 == m_world_size) {
                st.spread_ns += elapsed_ns(mb.first_arrival[p], t_arrive);
                st.last_count[rank].bump();
            }
            // This rank's own model work since it left the previous collective.
            if (st.last_exit[rank] != clk::time_point{}) {
                const uint64_t seg = elapsed_ns(st.last_exit[rank], t_arrive);
                st.seg_ns[rank].add(seg);
                st.seg_min_ns[rank].keep_min(seg);
                st.seg_max_ns[rank].keep_max(seg);
                st.seg_count[rank].bump();
            }
        }
        if (++mb.arrived[p] == m_world_size) {
            mb.arrived[p] = 0;
            mb.released[p] = mine.seq;
            mb.cv.notify_all();
        } else {
            const auto outcome = wait_for_condition(lk, mb.cv, m_collective_timeout, m_aborted, [&] {
                return mb.released[p] == mine.seq;
            });
            if (outcome != WaitOutcome::ready) {
                lk.unlock();
                fail_collective(outcome == WaitOutcome::timed_out, mine.collective_id, rank, "call rendezvous");
            }
        }
        all = mb.slots[p];
    }

    for (int q = 0; q < m_world_size; ++q) {
        const auto& s = all[static_cast<std::size_t>(q)];
        if (s.seq != mine.seq || s.allreduce != mine.allreduce || s.collective_id != mine.collective_id ||
            s.n != mine.n || s.rows != mine.rows || s.dtype != mine.dtype) {
            std::ostringstream oss;
            oss << "[TP][L0] ranks disagree on call " << mine.seq << ": rank " << rank << " is at collective "
                << mine.collective_id << " (n=" << mine.n << ", rows=" << mine.rows << ", " << mine.dtype
                << "), rank " << q << " at collective " << s.collective_id << " (call " << s.seq << ", n=" << s.n
                << ", rows=" << s.rows << ", " << s.dtype << ")";
            abort_all(oss.str());
            throw_if_aborted();
        }
    }
}

void GroupState::begin_allreduce(int rank,
                                 int collective_id,
                                 std::size_t n,
                                 ov::element::Type dtype,
                                 std::size_t half_bytes,
                                 AllreduceCall& call) {
    const Stopwatch staging_time(m_options.profiling_host);
    call.parity = static_cast<int>(m_allreduce_seq[static_cast<std::size_t>(rank)]++ & 1u);
    const std::size_t payload_bytes = checked_multiply(n, dtype.size(), "collective payload");
    void* const my_half = ensure_staging(m_staging[static_cast<std::size_t>(rank)],
                                         rank,
                                         payload_bytes,
                                         half_bytes,
                                         call.parity,
                                         collective_id);
    if (m_options.profiling_host) {
        m_stats.p2_gate_ns[rank].add(staging_time.elapsed().ns());
    }

    MailboxSlot mine;
    mine.seq = ++m_seq[static_cast<std::size_t>(rank)];
    mine.allreduce = true;
    mine.collective_id = collective_id;
    mine.n = n;
    mine.dtype = dtype;
    mine.staging = my_half;
    mine.staging_id = m_staging[static_cast<std::size_t>(rank)].id;
    std::vector<MailboxSlot> all;
    call_rendezvous(rank, mine, all);

    call.staging.resize(all.size());
    call.staging_ids.resize(all.size());
    for (std::size_t q = 0; q < all.size(); ++q) {
        call.staging[q] = all[q].staging;
        call.staging_ids[q] = all[q].staging_id;
    }
    call.recv = &m_events[call.parity].recv;
    call.step = &m_events[call.parity].step;
}

void GroupState::begin_gather(int rank,
                              int collective_id,
                              std::size_t rows,
                              std::size_t slice_elems,
                              ov::element::Type dtype,
                              void* out,
                              GatherCall& call) {
    call.parity = static_cast<int>(m_gather_seq[static_cast<std::size_t>(rank)]++ & 1u);

    MailboxSlot mine;
    mine.seq = ++m_seq[static_cast<std::size_t>(rank)];
    mine.allreduce = false;
    mine.collective_id = collective_id;
    mine.n = slice_elems;
    mine.rows = rows;
    mine.dtype = dtype;
    if (rank == 0) {
        mine.staging = out;
        mine.staging_id = alloc_id_of(m_shared->context, out);
    }
    std::vector<MailboxSlot> all;
    call_rendezvous(rank, mine, all);

    OPENVINO_ASSERT(all[0].staging != nullptr, "[TP][L0] gather ", collective_id, " has no destination on the root");
    call.root_out = all[0].staging;
    call.root_out_id = all[0].staging_id;
    call.events = &m_gather_events[call.parity].ev;
    call.ready = &m_gather_events[call.parity].ready;
}

void GroupState::track(Splice* splice) {
    std::lock_guard<std::mutex> lock(m_tracked_mutex);
    m_tracked.push_back(splice);
}

void GroupState::untrack(int rank, Splice* splice) {
    {
        std::lock_guard<std::mutex> lock(m_tracked_mutex);
        m_tracked.erase(std::remove(m_tracked.begin(), m_tracked.end(), splice), m_tracked.end());
    }
    auto& last = m_last_splice[static_cast<std::size_t>(rank)];
    if (last == splice) {
        last = nullptr;
    }
}

bool GroupState::await_splice(Splice& splice, int rank, int collective_id) {
    if (!splice.in_flight.load(std::memory_order_acquire)) {
        return false;
    }
    // Query before waiting: a memory read, where zeEventHostSynchronize costs
    // microseconds even with nothing to wait for.
    ze_result_t r = ze_api()->zeEventQueryStatus(splice.done);
    if (r == ZE_RESULT_NOT_READY) {
        r = ze_api()->zeEventHostSynchronize(splice.done, timeout_ns());
        if (r == ZE_RESULT_NOT_READY) {
            fail_collective(true, collective_id, rank, "previous splice of this recording");
        }
    }
    ZE_THROW(r);
    ZE_THROW(ze_api()->zeEventHostReset(splice.done));
    splice.in_flight.store(0, std::memory_order_release);
    return true;
}

void GroupState::handed_over(int rank, Splice* splice) {
    m_last_splice[static_cast<std::size_t>(rank)] = splice;
    note_collective_started(rank);
}

void GroupState::execute_and_wait(int rank, ze_command_list_handle_t list, const char* what) {
    auto& rs = m_ranks[static_cast<std::size_t>(rank)];
    ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(rs.compute_queue, 1, &list, nullptr));
    sync_queue(rs.compute_queue, what);
}

void GroupState::drain(int rank, const char* what) {
    auto& rs = m_ranks[static_cast<std::size_t>(rank)];
    if (rs.compute_queue) {
        sync_queue(rs.compute_queue, what);
    }
}

uint64_t GroupState::timeout_ns() const {
    if (m_collective_timeout.count() == 0) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(m_collective_timeout.count()) * 1'000'000ull;
}

void GroupState::sync_queue(ze_command_queue_handle_t queue, const char* what) const {
    const ze_result_t r = ze_api()->zeCommandQueueSynchronize(queue, timeout_ns());
    if (r == ZE_RESULT_NOT_READY) {
        OPENVINO_THROW("[TP][L0] ", what, " did not complete within ", m_collective_timeout.count(), " ms");
    }
    ZE_THROW(r);
}

void GroupState::abort_all(const std::string& reason) {
    {
        std::lock_guard<std::mutex> lock(m_abort_mutex);
        if (m_abort_reason.empty()) {
            m_abort_reason = reason;
        }
    }
    m_aborted.store(true, std::memory_order_release);

    // Under the mailbox lock: a rank that has just evaluated its predicate and
    // is about to sleep would otherwise miss the wakeup.
    std::lock_guard<std::mutex> lock(m_mailbox.mtx);
    m_mailbox.cv.notify_all();
}

void GroupState::throw_if_aborted() const {
    if (!m_aborted.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_abort_mutex);
    OPENVINO_THROW("[TP][L0] collective group is no longer usable: ", m_abort_reason);
}

void GroupState::fail_collective(bool timed_out, int collective_id, int rank, const char* stage) {
    if (timed_out) {
        std::ostringstream oss;
        oss << "[TP][L0] rank " << rank << " timed out after " << m_collective_timeout.count() << " ms at the "
            << stage << " of collective " << collective_id
            << "; the group is aborted. Either another rank never reached this collective or its "
               "device work never completed.";
        abort_all(oss.str());
    }
    throw_if_aborted();

    OPENVINO_THROW("[TP][L0] collective ", collective_id, " failed on rank ", rank, " at the ", stage);
}

void GroupState::note_collective_started(int rank) {
    std::call_once(m_watchdog_once, [this] {
        start_watchdog();
    });
    m_started[static_cast<std::size_t>(rank)].fetch_add(1, std::memory_order_release);
}

void GroupState::start_watchdog() {
    // A zero timeout asked for no deadline, so there is nothing to enforce.
    if (m_collective_timeout.count() <= 0) {
        return;
    }
    m_watchdog = std::thread([this] {
        watchdog_loop();
    });
}

void GroupState::stop_watchdog() {
    if (!m_watchdog.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_watchdog_mutex);
        m_watchdog_stop = true;
    }
    m_watchdog_cv.notify_all();
    m_watchdog.join();
}

void GroupState::watchdog_loop() {
    // Several polls within the deadline, or the reported timeout drifts well past it.
    const auto tick = std::max(std::chrono::milliseconds(50), m_collective_timeout / 4);

    // Progress is asked of the devices, not of the bookkeeping: the host runs
    // ahead on purpose, and a large prefill hands over every collective of a
    // pass before one is accounted for. A group that is merely behind keeps
    // turning completion events green; a stuck one stops, and so do the hosts.
    uint64_t seen = 0;
    auto since = std::chrono::steady_clock::now();

    for (;;) {
        {
            std::unique_lock<std::mutex> lock(m_watchdog_mutex);
            m_watchdog_cv.wait_for(lock, tick, [this] {
                return m_watchdog_stop;
            });
            if (m_watchdog_stop) {
                return;
            }
        }
        if (m_aborted.load(std::memory_order_acquire)) {
            return;
        }

        uint64_t signature = 0;
        uint64_t outstanding = 0;
        for (int r = 0; r < m_world_size; ++r) {
            signature += m_started[static_cast<std::size_t>(r)].load(std::memory_order_acquire);
        }
        {
            std::lock_guard<std::mutex> lock(m_tracked_mutex);
            for (const auto* splice : m_tracked) {
                if (!splice->in_flight.load(std::memory_order_acquire)) {
                    continue;
                }
                ++outstanding;
                if (ze_api()->zeEventQueryStatus(splice->done) == ZE_RESULT_SUCCESS) {
                    ++signature;
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        // Idle, or something moved since the last tick. A signature that goes
        // down counts too: a rank consumed an event and spliced again.
        if (outstanding == 0 || signature != seen) {
            seen = signature;
            since = now;
            continue;
        }
        if (now - since < m_collective_timeout) {
            continue;
        }

        std::ostringstream oss;
        oss << "[TP][L0] no collective has completed on any device for " << m_collective_timeout.count()
            << " ms while " << outstanding
            << " are outstanding. A rank's queue is waiting on a peer that never signalled.";
        abort_all(oss.str());
        release_all_waits();
        return;
    }
}

void GroupState::release_all_waits() {
    const auto& api = ze_api();
    auto signal = [&](ze_event_handle_t ev) {
        if (ev != nullptr) {
            api->zeEventHostSignal(ev);
        }
    };
    for (const auto& set : m_events) {
        for (auto ev : set.recv) {
            signal(ev);
        }
        for (auto ev : set.step) {
            signal(ev);
        }
    }
    for (const auto& set : m_gather_events) {
        for (auto ev : set.ev) {
            signal(ev);
        }
        for (auto ev : set.ready) {
            signal(ev);
        }
    }
    std::lock_guard<std::mutex> lock(m_tracked_mutex);
    for (const auto* splice : m_tracked) {
        signal(splice->done);
    }
}

void GroupState::note_collective_done(int rank) {
    if (!(m_options.profiling_host || m_options.profiling_device) || rank != 0) {
        return;
    }
    if (!m_dump_period.due()) {
        return;
    }
    emit_report();
}

void GroupState::emit_report() {
    const uint64_t calls = m_dump_period.calls();
    if (m_options.profiling_host) {
        report_host_ranks(calls);
        report_host_totals(calls);
    }
    if (m_options.profiling_device) {
        report_device(calls);
    }
}

void GroupState::report_host_ranks(uint64_t calls) {
    const auto& st = m_stats;
    const double c = static_cast<double>(calls);
    auto us = [c](uint64_t v) {
        return static_cast<double>(v) / 1.0e3 / c;
    };

    TP_REPORT << "[TP][RANK|host] calls=" << calls << " arrival spread=" << us(st.spread_ns) << "us/call"
              << std::endl;
    for (int r = 0; r < m_world_size; ++r) {
        const double last_share = 100.0 * static_cast<double>(st.last_count[r].get()) / c;
        const double sc = std::max<double>(1.0, static_cast<double>(st.seg_count[r].get()));
        TP_REPORT << "[TP][RANK|host]   rank " << r << ": late=" << us(st.late_ns[r].get()) << "us"
                  << " arrived_last=" << last_share << "%"
                  << "  ph1=" << us(st.ph1_ns[r].get()) << "us"
                  << " ph2=" << us(st.ph2_ns[r].get()) << "us"
                  << "  segment mean=" << (static_cast<double>(st.seg_ns[r].get()) / 1.0e3 / sc) << "us"
                  << " min=" << (static_cast<double>(st.seg_min_ns[r].get()) / 1.0e3)
                  << "us max=" << (static_cast<double>(st.seg_max_ns[r].get()) / 1.0e3) << "us" << std::endl;
        const uint64_t p2_known =
            st.p2_rec_ns[r].get() + st.p2_wait_ns[r].get() + st.p2_reset_ns[r].get() + st.p2_append_ns[r].get();
        TP_REPORT << "[TP][RANK|host]     ph1 staging check=" << us(st.p2_gate_ns[r].get()) << "us"
                  << "  ph2 split: record=" << us(st.p2_rec_ns[r].get()) << "us"
                  << "(" << st.p2_rec_count[r].get() << "x)"
                  << " evt_wait=" << us(st.p2_wait_ns[r].get()) << "us"
                  << " evt_reset=" << us(st.p2_reset_ns[r].get()) << "us"
                  << "(" << st.p2_wait_count[r].get() << "x, blocked " << st.p2_block_count[r].get() << "x)"
                  << " append=" << us(st.p2_append_ns[r].get()) << "us"
                  << " rest=" << us(st.ph2_ns[r].get() - std::min(p2_known, st.ph2_ns[r].get())) << "us"
                  << std::endl;
        const uint64_t ahead_n = st.ahead_count[r].get();
        if (ahead_n > 0) {
            TP_REPORT << "[TP][RANK|host]     run-ahead: mean="
                      << static_cast<double>(st.ahead_sum[r].get()) / static_cast<double>(ahead_n)
                      << " splices max=" << st.ahead_max[r].get() << std::endl;
        }
    }
}

void GroupState::report_host_totals(uint64_t calls) {
    const auto& st = m_stats;
    const double c = static_cast<double>(calls);
    auto tms = [](uint64_t ns) {
        return static_cast<double>(ns) / 1.0e6;
    };
    TP_REPORT << "[TP][TOTAL|host] r0 calls=" << calls << " rebuilds=" << st.rebuilds.get()
              << " records=" << st.records.get() << "  totals: ph1=" << tms(st.ph1_ns[0].get()) << "ms"
              << " ph2=" << tms(st.ph2_ns[0].get()) << "ms"
              << " (record=" << tms(st.record_ns.get()) << "ms"
              << ", splice=" << tms(st.splice_ns.get()) << "ms)"
              << "  per-call: ph1=" << tms(st.ph1_ns[0].get()) / c << "ms"
              << " prep=" << tms(st.prep_ns.get()) / c << "ms"
              << " splice=" << tms(st.splice_ns.get()) / c << "ms"
              << " tail=" << tms(st.tail_ns.get()) / c << "ms" << std::endl;
    uint64_t pair = 0, ring = 0, halving = 0;
    for (int r = 0; r < m_world_size; ++r) {
        pair += st.n_pair[r].get();
        ring += st.n_ring[r].get();
        halving += st.n_halving[r].get();
    }
    TP_REPORT << "[TP][TOTAL|host]   schedule of " << (pair + ring + halving) << " recordings: pair=" << pair
              << " ring=" << ring << " halving=" << halving << std::endl;
    // Divided by recordings, not calls: recording is rare but lands in time to first token.
    const double rc = std::max<double>(1.0, static_cast<double>(st.records.get()));
    auto rec_ms = [](const Counter& v) {
        return static_cast<double>(v.get()) / 1.0e6;
    };
    TP_REPORT << "[TP][TOTAL|host]   record breakdown: records=" << st.records.get()
              << " per-record=" << tms(st.record_ns.get()) / rc << "ms"
              << " (drain=" << rec_ms(st.rec_drain_ns) / rc << "ms"
              << " reset=" << rec_ms(st.rec_reset_ns) / rc << "ms"
              << " append=" << (rec_ms(st.rec_build_ns) - rec_ms(st.rec_close_ns)) / rc << "ms"
              << " close=" << rec_ms(st.rec_close_ns) / rc << "ms)" << std::endl;
    const auto scratch = get_scratch_stats();
    TP_REPORT << "[TP][TOTAL|host]   scratch: payload_capacity=" << (scratch.payload_capacity_bytes / (1024.0 * 1024.0))
              << "MB"
              << " total=" << (scratch.total_allocated_bytes / (1024.0 * 1024.0)) << "MB"
              << " generation=" << scratch.generation << " grows=" << scratch.growth_count
              << " allocations=" << scratch.allocation_count
              << " stall=" << (static_cast<double>(st.scratch_stall_ns.get()) / 1.0e6) << "ms" << std::endl;
    for (int r = 0; r < m_world_size; ++r) {
        const uint64_t gc_calls = st.gather_calls[r].get();
        if (gc_calls == 0) {
            continue;
        }
        const auto gus = [gc_calls](uint64_t v) {
            return static_cast<double>(v) / 1.0e3 / static_cast<double>(gc_calls);
        };
        TP_REPORT << "[TP][TOTAL|host]   gather rank " << r << ": calls=" << gc_calls
                  << " spliced=" << st.gather_spliced[r].get() << " records=" << st.gather_records[r].get()
                  << " barrier=" << gus(st.gather_barrier_ns[r].get()) << "us"
                  << " append=" << gus(st.gather_append_ns[r].get()) << "us"
                  << " close=" << (static_cast<double>(st.gather_close_ns.get()) / 1.0e6) << "ms total"
                  << std::endl;
    }
}

void GroupState::report_device(uint64_t calls) {
    const auto& st = m_stats;
    const double c = static_cast<double>(calls);
    uint64_t total_samples = 0;
    for (int r = 0; r < m_world_size; ++r) {
        total_samples += st.p2_dev_count[r].get();
    }
    if (total_samples == 0) {
        // A completion event is read when the same list is spliced again, two
        // instances later, so the first numbers appear on the third instance.
        TP_REPORT << "[TP][RANK|dev] calls=" << calls
                  << ": no samples yet -- device timings start after a"
                     " collective has run three times"
                  << std::endl;
        return;
    }

    TP_REPORT << "[TP][RANK|dev] calls=" << calls << " schedule=" << (m_world_size > 2 ? "ring/halving" : "pair")
              << " world=" << m_world_size << std::endl;
    uint64_t all_bytes = 0;
    uint64_t all_ns = 0;
    for (int r = 0; r < m_world_size; ++r) {
        const uint64_t samples = st.p2_dev_count[r].get();
        if (samples == 0) {
            continue;
        }
        const uint64_t ns = st.p2_dev_ns[r].get();
        const uint64_t bytes = st.dev_bytes[r].get();
        all_bytes += bytes;
        all_ns += ns;
        const double busy_us = static_cast<double>(ns) / 1.0e3 / static_cast<double>(samples);
        const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0) / static_cast<double>(samples);
        // Bytes over queue occupancy: the rate the model sees, kernel and peer waits included.
        const double gbs =
            ns > 0 ? (static_cast<double>(bytes) / 1.0e9) / (static_cast<double>(ns) / 1.0e9) : 0.0;
        TP_REPORT << "[TP][RANK|dev]   rank " << r << ": queue busy mean=" << busy_us << "us"
                  << " max=" << (static_cast<double>(st.p2_dev_max_ns[r].get()) / 1.0e3) << "us(cid "
                  << st.p2_dev_max_cid[r].get() << ")"
                  << "  sent=" << mb << "MB/call"
                  << " effective=" << gbs << " GB/s"
                  << " (" << samples << " samples)" << std::endl;

        // Percentiles off the histogram, as bucket upper bounds.
        auto pct_us = [&](double frac) -> uint64_t {
            const auto want = static_cast<uint64_t>(static_cast<double>(samples) * frac);
            uint64_t seen = 0;
            for (int b = 0; b < CollectiveStats::kDevBuckets; ++b) {
                seen += st.p2_dev_hist[static_cast<std::size_t>(r) *
                                           static_cast<std::size_t>(CollectiveStats::kDevBuckets) +
                                       static_cast<std::size_t>(b)]
                            .get();
                if (seen >= want) {
                    return uint64_t{1} << (b + 1);
                }
            }
            return uint64_t{1} << CollectiveStats::kDevBuckets;
        };
        TP_REPORT << "[TP][RANK|dev]     queue busy under: p50=" << pct_us(0.50) << "us"
                  << " p90=" << pct_us(0.90) << "us"
                  << " p99=" << pct_us(0.99) << "us" << std::endl;

        const uint64_t gaps = st.dev_gap_count[r].get();
        if (gaps > 0) {
            const double gap_us = static_cast<double>(st.dev_gap_ns[r].get()) / 1.0e3 / static_cast<double>(gaps);
            const double duty = 100.0 * busy_us / (busy_us + gap_us);
            TP_REPORT << "[TP][RANK|dev]     between collectives: model=" << gap_us << "us"
                      << " -> collectives own " << duty << "% of device time"
                      << " (" << gaps << " gaps)" << std::endl;
        }
    }
    if (all_ns > 0) {
        TP_REPORT << "[TP][TOTAL|dev] across " << m_world_size
                  << " ranks: sent=" << (static_cast<double>(all_bytes) / (1024.0 * 1024.0) / c) << "MB/call"
                  << " aggregate="
                  << ((static_cast<double>(all_bytes) / 1.0e9) / (static_cast<double>(all_ns) / 1.0e9 / m_world_size))
                  << " GB/s"
                  << "  (time is queue occupancy, kernel and peer waits included)" << std::endl;
    }
    for (int r = 0; r < m_world_size; ++r) {
        const uint64_t gd = st.gather_dev_count[r].get();
        if (gd == 0) {
            continue;
        }
        const double busy_us = static_cast<double>(st.gather_dev_ns[r].get()) / 1.0e3 / static_cast<double>(gd);
        const double mb = static_cast<double>(st.gather_bytes[r].get()) / (1024.0 * 1024.0) / static_cast<double>(gd);
        TP_REPORT << "[TP][RANK|dev]   gather rank " << r << ": queue busy mean=" << busy_us << "us"
                  << " max=" << (static_cast<double>(st.gather_dev_max_ns[r].get()) / 1.0e3)
                  << "us  sent=" << mb << "MB/call"
                  << " (" << gd << " samples)" << std::endl;
    }
}

}  // namespace tp
}  // namespace ze
}  // namespace cldnn
