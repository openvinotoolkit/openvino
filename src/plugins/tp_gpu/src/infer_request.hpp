// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace tp_gpu {

class CompiledModel;

/// \brief Inference request for single-graph tensor parallel.
///
/// Each rank has one compiled model (containing in-graph TPAllReduce ops).
/// Execution: set user inputs on all ranks → launch all ranks in parallel →
/// collect output from rank 0.
/// AllReduce synchronization happens inside each GPU's inference pipeline
/// via the TPAllReduce CPU primitive and shared TPDeviceCoordinator object.
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);

    void infer() override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    std::shared_ptr<const CompiledModel> m_compiled_model;

    /// One infer request per rank.
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> m_rank_requests;

    /// Cached fan-out wrappers built on first query_state() call; same
    /// vector is returned on every subsequent call to keep object identity
    /// stable for downstream consumers and to avoid repeatedly grouping
    /// per-rank states.
    mutable std::mutex m_state_mutex;
    mutable std::vector<ov::SoPtr<ov::IVariableState>> m_fanout_states;
};

}  // namespace tp_gpu
}  // namespace ov
