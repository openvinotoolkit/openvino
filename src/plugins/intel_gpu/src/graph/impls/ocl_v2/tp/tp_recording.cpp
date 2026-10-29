// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "tp_recording.hpp"

#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::ze::tp::ze_api;

TPRecording::~TPRecording() {
    if (!m_group) {
        return;
    }
    // A list or event the device may still touch must not be destroyed; one
    // that does not finish within the timeout is leaked instead.
    bool idle = true;
    for (int p = 0; p < 2; ++p) {
        auto& s = m_splices[p];
        if (s.in_flight.load(std::memory_order_acquire) &&
            ze_api()->zeEventHostSynchronize(s.done, m_group->timeout_ns()) != ZE_RESULT_SUCCESS) {
            idle = false;
        }
        m_group->untrack(m_rank, &s);
    }
    if (idle) {
        cldnn::ze::tp::GroupState::destroy_recording_resources(m_res);
    }
}

void TPRecording::bind(const cldnn::ze::tp::Channel& channel, int collective_id, bool allreduce) {
    if (m_group) {
        OPENVINO_ASSERT(m_group == channel.group() && m_rank == channel.rank(),
                        "[TP][L0] a collective recording cannot move to another channel");
        return;
    }
    m_res = channel.group()->take_recording_resources(collective_id, channel.rank());
    m_splices = std::make_unique<cldnn::ze::tp::Splice[]>(2);
    m_group = channel.group();
    m_rank = channel.rank();
    for (int p = 0; p < 2; ++p) {
        m_splices[p].done = m_res.done[p];
        m_splices[p].allreduce = allreduce;
        m_group->track(&m_splices[p]);
    }
}

void TPRecording::reset(int parity, bool timed) {
    auto& st = m_group->stats();
    const bool measure = timed && m_group->options().profiling_host;
    {
        cldnn::ze::tp::ScopedTime drain(measure, st.rec_drain_ns);
        m_group->drain(m_rank, "re-record: compute queue drain");
    }
    cldnn::ze::tp::ScopedTime reset(measure, st.rec_reset_ns);
    OV_ZE_EXPECT(ze_api()->zeCommandListReset(m_res.lists[parity]));
}

void TPRecording::splice_into(ze_command_list_handle_t model_queue, int parity) {
    // Up before the driver call, so a block inside it still looks outstanding to the watchdog.
    m_splices[parity].in_flight.store(1, std::memory_order_release);
    OV_ZE_EXPECT(ze_api()->zeCommandListImmediateAppendCommandListsExp(model_queue,
                                                                      1,
                                                                      &m_res.lists[parity],
                                                                      m_splices[parity].done,
                                                                      0,
                                                                      nullptr));
    m_group->handed_over(m_rank, &m_splices[parity]);
}

}  // namespace ov::intel_gpu::ocl

#endif  // ENABLE_TP
