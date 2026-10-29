// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "ze/tp/rank_communicator.hpp"

#include <memory>

namespace ov::intel_gpu::ocl {

/// The command lists of one collective on one rank, one per call parity, and
/// their hand-over to the device. Belongs to one primitive instance: a copy
/// starts empty and binds on its own first use.
class TPRecording {
public:
    TPRecording() = default;
    TPRecording(const TPRecording&) : TPRecording() {}
    TPRecording& operator=(const TPRecording&) = delete;
    ~TPRecording();

    /// Takes over the lists and completion events the group made for this
    /// collective and rank on first use, and registers the events with it.
    void bind(const cldnn::ze::tp::Channel& channel, int collective_id, bool allreduce);

    cldnn::ze::tp::GroupState& group() const {
        return *m_group;
    }
    int rank() const {
        return m_rank;
    }
    ze_command_list_handle_t list(int parity) const {
        return m_res.lists[parity];
    }
    cldnn::ze::tp::Splice& splice(int parity) const {
        return m_splices[parity];
    }

    /// Clears list(parity) for re-recording. Its previous splice must have
    /// been consumed: resetting a list the device still runs is undefined.
    /// `timed` books the time into the allreduce record breakdown.
    void reset(int parity, bool timed);

    /// Appends list(parity) to the model queue, signalling its splice on completion.
    void splice_into(ze_command_list_handle_t model_queue, int parity);

private:
    cldnn::ze::tp::GroupStatePtr m_group;
    int m_rank{-1};
    cldnn::ze::tp::GroupState::RecordingResources m_res;
    std::unique_ptr<cldnn::ze::tp::Splice[]> m_splices;
};

}  // namespace ov::intel_gpu::ocl
