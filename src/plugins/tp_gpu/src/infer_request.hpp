// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_set>
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

    /// The caller is not expected to supply the paged-attention cache: that
    /// memory belongs to the plugin, which binds it per rank.  Every other
    /// port still has to carry a tensor before an inference starts.
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    /// Hand each rank the slice of the paged-attention cache it owns.  A no-op
    /// when the model has no such cache, or when the tensors already bound are
    /// still the current ones.
    void bind_cache();

    /// Returns the tensor to hand rank `rank` for user input `input_idx`, or a
    /// null pointer to hand the caller's tensor unchanged.
    ///
    /// A plain host tensor forces the GPU plugin to stage every input through
    /// a device copy, and for the inputs a CPU implementation reads it makes
    /// that copy blocking: measured at 4 us alone but 500 us once four ranks
    /// share the process, three times per token.  A tensor that already lives
    /// in the rank's context is taken as-is instead, with no copy and no wait.
    ///
    /// Only small inputs are staged this way.  The wait it removes is a fixed
    /// cost, so it dominates exactly when the payload is tiny, while a large
    /// input is better left in device memory -- the GPU plugin itself avoids
    /// USM host for big buffers on discrete cards.
    ov::SoPtr<ov::ITensor> stage_input(size_t rank,
                                       size_t input_idx,
                                       const ov::SoPtr<ov::ITensor>& user_tensor);

    std::shared_ptr<const CompiledModel> m_compiled_model;

    /// One infer request per rank.
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> m_rank_requests;

    /// Cached fan-out wrappers built on first query_state() call; same
    /// vector is returned on every subsequent call to keep object identity
    /// stable for downstream consumers and to avoid repeatedly grouping
    /// per-rank states.
    mutable std::mutex m_state_mutex;
    mutable std::vector<ov::SoPtr<ov::IVariableState>> m_fanout_states;

    /// User inputs that name a paged-attention cache port.  They are filled
    /// from the cache controller, not from the caller's tensors.
    std::unordered_set<size_t> m_cache_input_indices;

    /// Cache generation the rank requests were last bound to.
    uint64_t m_bound_cache_generation = 0;

    /// Plugin-owned USM-host staging for small user inputs, indexed
    /// [rank][user input index].  Empty entries mean "not staged": either the
    /// input is too large, or the caller already supplies device-side memory,
    /// or the rank's context refused to allocate.
    std::vector<std::vector<ov::SoPtr<ov::ITensor>>> m_input_stage;
    /// Bytes actually allocated per staging tensor.  Kept apart from the
    /// tensor's own size because set_shape() shrinks that, and a port that
    /// alternates between a long prompt and a single token would otherwise
    /// reallocate on every switch.
    std::vector<std::vector<size_t>> m_input_stage_capacity;
    /// Ports whose staging was tried and failed; never retried.
    std::vector<std::vector<uint8_t>> m_input_stage_refused;
};

}  // namespace tp_gpu
}  // namespace ov
