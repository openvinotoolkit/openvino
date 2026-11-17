// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "tp_gather_recorder.hpp"
#include "tp_common.hpp"

#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "intel_gpu/runtime/itt.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::ze::tp::alloc_id_of;
using cldnn::ze::tp::ScopedTime;
using cldnn::ze::tp::Stopwatch;
using cldnn::ze::tp::ze_api;

void TPGatherRecorder::record(uint32_t parity) {
    const Signature& sig = m_sig[parity];
    auto& group = m_rec.group();
    auto& ev = *m_call.events;
    auto& ready = *m_call.ready;
    const uint32_t rank = m_rec.rank();
    const uint32_t world_size = group.world_size();
    const std::size_t slice_bytes = checked_multiply(sig.slice_elems, sig.dtype.size(), "gather slice");
    const std::size_t full_bytes = checked_multiply(slice_bytes, world_size, "gather row");

    m_rec.reset(parity, false);
    ze_command_list_handle_t list = m_rec.list(parity);

    // This rank's band of the root's rows, a copy per row: a region copy left the first bytes of a
    // row's band stale on the root about once in 500 calls.
    auto copy_band = [&] {
        for (std::size_t row = 0; row < sig.rows; ++row) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                                 static_cast<uint8_t*>(sig.root_out) + row * full_bytes +
                                                                     rank * slice_bytes,
                                                                 static_cast<const uint8_t*>(sig.in) + row * slice_bytes,
                                                                 slice_bytes,
                                                                 nullptr,
                                                                 0,
                                                                 nullptr));
        }
    };

    if (rank != 0) {
        // Not before the root's queue has reached this gather: until then its
        // output's memory may still be in use by whatever ran there before.
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ready[rank]));
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendEventReset(list, ready[rank]));
        copy_band();
        // Signalled once the reset and the copy are both done, so the root
        // cannot hand out the next ready of this parity before it is cleared.
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendBarrier(list, ev[rank], 0, nullptr));
    } else {
        // Own band first: the barrier and the device-scope signals after it
        // write its cache lines back before any peer writes next to them.
        copy_band();
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
        for (uint32_t r = 1; r < world_size; ++r) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendSignalEvent(list, ready[r]));
        }
        // Holds everything queued behind this recording -- the rest of the
        // model on the root -- until every band is in place.
        std::vector<ze_event_handle_t> waits(ev.begin() + 1, ev.end());
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, world_size - 1, waits.data()));
        for (auto* e : waits) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendEventReset(list, e));
        }
    }

    ScopedTime close(group.options().profiling_host, group.stats().gather_close_ns);
    OV_ZE_EXPECT(ze_api()->zeCommandListClose(list));
}

void TPGatherRecorder::run(const cldnn::ze::tp::Channel& channel,
                           uint64_t collective_id,
                           const std::string& name,
                           void* in,
                           void* out,
                           std::size_t rows,
                           std::size_t slice_elems,
                           ov::element::Type dtype,
                           ze_command_list_handle_t model_queue) {
    auto& group = *channel.group();
    const uint32_t rank = channel.rank();
    OPENVINO_ASSERT(in != nullptr, "[GPU][TP][L0] gather called with a null source on rank ", rank);
    OPENVINO_ASSERT(rank != 0 || out != nullptr, "[GPU][TP][L0] gather called with a null destination on the root rank");
    OPENVINO_ASSERT(rows > 0 && slice_elems > 0,
                    "[GPU][TP][L0] gather of an empty slice (rows=",
                    rows,
                    ", slice=",
                    slice_elems,
                    ")");
    group.throw_if_aborted();
    m_rec.bind(channel, false);

    auto& st = group.stats();
    const bool g_host = group.options().profiling_host;
    const bool g_dev = group.options().profiling_device;
    const Stopwatch since_entry(g_host);

    {
        OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_gather_rendezvous");
        group.begin_gather(rank, collective_id, rows, slice_elems, dtype, out, m_call);
    }
    const uint32_t parity = m_call.parity;
    if (g_host) {
        st.gather_barrier_ns[rank].add(since_entry.elapsed().ns());
        st.gather_calls[rank].bump();
    }

    AbortOnFailure abort_guard{group, collective_id, name, rank, "gathering"};

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
        if (group.wait_splice(splice, rank, collective_id)) {
            if (g_dev) {
                // The event holds the previous instance's timestamps until the release.
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
            group.release_splice(splice, rank);
        }
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
