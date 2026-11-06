// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "tp_gather_recorder.hpp"

#include <limits>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "intel_gpu/runtime/itt.hpp"

#define ZE_THROW(expr) OV_ZE_EXPECT(expr)

namespace ov::intel_gpu::ocl {

using cldnn::ze::tp::alloc_id_of;
using cldnn::ze::tp::ScopedTime;
using cldnn::ze::tp::Stopwatch;
using cldnn::ze::tp::ze_api;

namespace {

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* what) {
    OPENVINO_ASSERT(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
                    "[TP][L0] ",
                    what,
                    " byte count overflow: ",
                    lhs,
                    " * ",
                    rhs);
    return lhs * rhs;
}

uint64_t ticks_to_ns(uint64_t start, uint64_t end, uint64_t mask, uint64_t ns_per_tick) {
    const uint64_t s = start & mask;
    const uint64_t e = end & mask;
    const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
    return delta * ns_per_tick;
}

}  // namespace

void TPGatherRecorder::record(int parity) {
    const Signature& sig = m_sig[parity];
    auto& group = m_rec.group();
    auto& ev = *m_call.events;
    auto& ready = *m_call.ready;
    const int rank = m_rec.rank();
    const int world_size = group.world_size();
    const std::size_t slice_bytes = checked_multiply(sig.slice_elems, sig.dtype.size(), "gather slice");
    const std::size_t full_bytes = checked_multiply(slice_bytes, static_cast<std::size_t>(world_size), "gather row");

    OPENVINO_ASSERT(full_bytes <= std::numeric_limits<uint32_t>::max(),
                    "[TP][L0] gather row of ",
                    full_bytes,
                    " bytes exceeds the pitch a Level Zero region copy can express");
    OPENVINO_ASSERT(sig.rows <= std::numeric_limits<uint32_t>::max(),
                    "[TP][L0] gather of ",
                    sig.rows,
                    " rows exceeds the region copy height");

    m_rec.reset(parity, false);
    ze_command_list_handle_t list = m_rec.list(parity);

    // This rank's band of the root's rows: one contiguous run for a single row,
    // otherwise one region copy rather than `rows` transfers.
    auto copy_band = [&] {
        if (sig.rows == 1) {
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             static_cast<uint8_t*>(sig.root_out) +
                                                                 static_cast<std::size_t>(rank) * slice_bytes,
                                                             sig.in,
                                                             slice_bytes,
                                                             nullptr,
                                                             0,
                                                             nullptr));
            return;
        }
        ze_copy_region_t dst_region{};
        dst_region.originX = static_cast<uint32_t>(static_cast<std::size_t>(rank) * slice_bytes);
        dst_region.width = static_cast<uint32_t>(slice_bytes);
        dst_region.height = static_cast<uint32_t>(sig.rows);
        dst_region.depth = 1;
        ze_copy_region_t src_region{};
        src_region.width = static_cast<uint32_t>(slice_bytes);
        src_region.height = static_cast<uint32_t>(sig.rows);
        src_region.depth = 1;
        ZE_THROW(ze_api()->zeCommandListAppendMemoryCopyRegion(list,
                                                               sig.root_out,
                                                               &dst_region,
                                                               static_cast<uint32_t>(full_bytes),
                                                               0,
                                                               sig.in,
                                                               &src_region,
                                                               static_cast<uint32_t>(slice_bytes),
                                                               0,
                                                               nullptr,
                                                               0,
                                                               nullptr));
    };

    if (rank != 0) {
        // Not before the root's queue has reached this gather: until then its
        // output's memory may still be in use by whatever ran there before.
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ready[static_cast<std::size_t>(rank)]));
        ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ready[static_cast<std::size_t>(rank)]));
        copy_band();
        // Signalled once the reset and the copy are both done, so the root
        // cannot hand out the next ready of this parity before it is cleared.
        ZE_THROW(ze_api()->zeCommandListAppendBarrier(list, ev[static_cast<std::size_t>(rank)], 0, nullptr));
    } else {
        // Own band first: the barrier and the device-scope signals after it
        // write its cache lines back before any peer writes next to them.
        copy_band();
        ZE_THROW(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
        for (int r = 1; r < world_size; ++r) {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ready[static_cast<std::size_t>(r)]));
        }
        // Holds everything queued behind this recording -- the rest of the
        // model on the root -- until every band is in place.
        std::vector<ze_event_handle_t> waits(ev.begin() + 1, ev.end());
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, static_cast<uint32_t>(waits.size()), waits.data()));
        for (auto* e : waits) {
            ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, e));
        }
    }

    ScopedTime close(group.options().profiling_host, group.stats().gather_close_ns);
    ZE_THROW(ze_api()->zeCommandListClose(list));
}

void TPGatherRecorder::run(const cldnn::ze::tp::Channel& channel,
                           int collective_id,
                           void* in,
                           void* out,
                           std::size_t rows,
                           std::size_t slice_elems,
                           ov::element::Type dtype,
                           ze_command_list_handle_t model_queue) {
    auto& group = *channel.group();
    const int rank = channel.rank();
    OPENVINO_ASSERT(collective_id >= 0 && collective_id < group.num_collectives(),
                    "[TP][L0] collective_id out of range: ",
                    collective_id);
    OPENVINO_ASSERT(in != nullptr, "[TP][L0] gather called with a null source on rank ", rank);
    OPENVINO_ASSERT(rank != 0 || out != nullptr, "[TP][L0] gather called with a null destination on the root rank");
    OPENVINO_ASSERT(rows > 0 && slice_elems > 0,
                    "[TP][L0] gather of an empty slice (rows=",
                    rows,
                    ", slice=",
                    slice_elems,
                    ")");
    group.throw_if_aborted();
    m_rec.bind(channel, collective_id, false);

    auto& st = group.stats();
    const bool g_host = group.options().profiling_host;
    const bool g_dev = group.options().profiling_device;
    const Stopwatch since_entry(g_host);

    {
        OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_gather_rendezvous");
        group.begin_gather(rank, collective_id, rows, slice_elems, dtype, out, m_call);
    }
    const int parity = m_call.parity;
    if (g_host) {
        st.gather_barrier_ns[rank].add(since_entry.elapsed().ns());
        st.gather_calls[rank].bump();
    }

    struct AbortOnFailure {
        cldnn::ze::tp::GroupState& group;
        int collective_id;
        int rank;
        bool armed{true};
        ~AbortOnFailure() {
            if (armed) {
                group.abort_all("[TP][L0] rank " + std::to_string(rank) + " failed while gathering collective " +
                                std::to_string(collective_id));
            }
        }
    } abort_guard{group, collective_id, rank};

    Signature sig;
    sig.valid = true;
    sig.in = in;
    sig.in_id = alloc_id_of(group.context(), in);
    if (rank == 0) {
        sig.out = out;
        sig.out_id = alloc_id_of(group.context(), out);
    }
    sig.rows = rows;
    sig.slice_elems = slice_elems;
    sig.dtype = dtype;
    sig.root_out = m_call.root_out;
    sig.root_out_id = m_call.root_out_id;

    // Runs only once the previous splice of the list has been consumed.
    auto record_if_stale = [&] {
        if (m_sig[parity] == sig) {
            return;
        }
        m_sig[parity] = sig;
        record(parity);
        if (g_host) {
            st.gather_records[rank].bump();
        }
    };

    auto& splice = m_rec.splice(parity);
    if (model_queue != nullptr && channel.run_spliced()) {
        // Read the previous instance's timestamps before await_splice clears
        // them. Blocking: with the host far ahead, a query would skip almost
        // every sample.
        if (g_dev && splice.in_flight.load(std::memory_order_acquire)) {
            if (ze_api()->zeEventHostSynchronize(splice.done, group.timeout_ns()) == ZE_RESULT_SUCCESS) {
                ze_kernel_timestamp_result_t kt{};
                if (ze_api()->zeEventQueryKernelTimestamp(splice.done, &kt) == ZE_RESULT_SUCCESS) {
                    const uint64_t ns = ticks_to_ns(kt.global.kernelStart,
                                                    kt.global.kernelEnd,
                                                    group.timestamp_mask(rank),
                                                    group.timer_ns_per_tick(rank));
                    st.gather_dev_ns[rank].add(ns);
                    st.gather_dev_max_ns[rank].keep_max(ns);
                    st.gather_dev_count[rank].bump();
                    st.gather_bytes[rank].add(splice.bytes);
                }
            }
        }
        group.await_splice(splice, rank, collective_id);
        record_if_stale();
        const Stopwatch splice_time(g_host);
        {
            OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_gather_append");
            m_rec.splice_into(model_queue, parity);
        }
        if (g_host) {
            st.gather_append_ns[rank].add(splice_time.elapsed().ns());
            st.gather_spliced[rank].bump();
        }
        if (g_dev) {
            splice.bytes = checked_multiply(checked_multiply(rows, slice_elems, "gather slice"),
                                            dtype.size(),
                                            "gather bytes");
        }
    } else {
        record_if_stale();
        group.execute_and_wait(rank, m_rec.list(parity), "gather: own queue");
    }
    abort_guard.armed = false;
    group.note_collective_done(rank);
}

}  // namespace ov::intel_gpu::ocl

#endif  // ENABLE_TP
