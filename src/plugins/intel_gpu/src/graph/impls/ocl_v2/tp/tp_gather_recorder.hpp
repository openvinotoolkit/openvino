// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tp_recording.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::ocl {

/// \brief The Level Zero gather of one TPGather on one rank: each rank's
/// `rows` x `slice_elems` matrix lands as a column band of the root's
/// `rows` x (`slice_elems` * world_size) buffer. Only rank 0 is written -- the
/// infer request reads outputs from rank 0 alone.
class TPGatherRecorder {
public:
    TPGatherRecorder() = default;
    TPGatherRecorder(const TPGatherRecorder&) : TPGatherRecorder() {}
    TPGatherRecorder& operator=(const TPGatherRecorder&) = delete;

    /// `out` is the destination on rank 0 and ignored elsewhere. `model_queue`
    /// works as for TPAllReduceRecorder::run(); spliced, the root's recording
    /// waits for every other rank's slice, which holds the root's model queue
    /// until the buffer is whole.
    void run(const cldnn::ze::tp::Channel& channel,
             int collective_id,
             void* in,
             void* out,
             std::size_t rows,
             std::size_t slice_elems,
             ov::element::Type dtype,
             ze_command_list_handle_t model_queue);

private:
    struct Signature {
        bool valid{false};
        void* in{nullptr};
        uint64_t in_id{0};
        std::size_t rows{0};
        std::size_t slice_elems{0};
        ov::element::Type dtype{ov::element::dynamic};
        void* root_out{nullptr};
        uint64_t root_out_id{0};

        bool operator==(const Signature& o) const {
            return valid == o.valid && in == o.in && in_id == o.in_id && rows == o.rows &&
                   slice_elems == o.slice_elems && dtype == o.dtype && root_out == o.root_out &&
                   root_out_id == o.root_out_id;
        }
    };

    void record(int parity, std::vector<ze_event_handle_t>& events);

    TPRecording m_rec;
    Signature m_sig[2];
    cldnn::ze::tp::GroupState::GatherCall m_call;
};

}  // namespace ov::intel_gpu::ocl
