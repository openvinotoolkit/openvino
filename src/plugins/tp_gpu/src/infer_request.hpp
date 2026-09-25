// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/tensor.hpp"
#include "tp_gpu/tp_debug.hpp"

namespace ov {
namespace tp_gpu {

class CompiledModel;

/// \brief Inference request for single-graph tensor parallel.
///
/// Every rank has a compiled model of its own, with the collectives already in
/// the graph. An inference sets the user's inputs on all of them, runs them in
/// parallel and reads the outputs back from rank 0.
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const CompiledModel>& compiled_model);

    void infer() override;

    /// The caller is not expected to supply the paged-attention cache: that
    /// memory belongs to the plugin, which binds it per rank. Every other
    /// port still has to carry a tensor before an inference starts.
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    /// Hand each rank the slice of the paged-attention cache it owns. A no-op
    /// when the model has no such cache, or when the tensors already bound are
    /// still the current ones.
    void bind_cache();

    /// Give every rank the caller's inputs, staging the small ones.
    void set_rank_inputs();

    /// Run all ranks and wait for them.
    void run_ranks();

    /// Publish rank 0's outputs as this request's own.
    void collect_outputs();

    /// Returns the tensor to hand rank `rank` for user input `input_idx`, or a
    /// null pointer to hand the caller's tensor unchanged.
    ov::SoPtr<ov::ITensor> stage_input(size_t rank,
                                       size_t input_idx,
                                       const ov::SoPtr<ov::ITensor>& user_tensor);

    std::shared_ptr<const CompiledModel> m_compiled_model;

    /// One infer request per rank.
    std::vector<ov::SoPtr<ov::IAsyncInferRequest>> m_rank_requests;

    /// Cached fan-out wrappers, built on the first query_state().
    /// Returning the same vector every time keeps object identity stable for downstream
    /// consumers and avoids re-grouping the per-rank states.
    mutable std::mutex m_state_mutex;
    mutable std::vector<ov::SoPtr<ov::IVariableState>> m_fanout_states;

    /// Per user input: whether it names a paged-attention cache port.
    /// Those are filled from the cache controller, not from the caller's tensors.
    std::vector<uint8_t> m_cache_input;

    /// Cache generation the rank requests were last bound to.
    uint64_t m_bound_cache_generation = 0;

    /// Size ceiling for staging an input through plugin-owned host memory.
    size_t m_stage_limit = 0;

    /// Plugin-owned USM-host staging for small user inputs.
    /// Empty entries mean "not staged": either the input is too large,
    /// or the caller already supplies device-side memory,
    /// or the rank's context refused to allocate.
    std::vector<std::vector<ov::SoPtr<ov::ITensor>>> m_input_stage;
    /// Bytes actually allocated per staging tensor. Kept apart from the
    /// tensor's own size because set_shape() shrinks that, and a port that
    /// alternates between a long prompt and a single token would otherwise
    /// reallocate on every switch.
    std::vector<std::vector<size_t>> m_input_stage_capacity;
    /// Ports whose staging was tried and failed; never retried.
    std::vector<std::vector<uint8_t>> m_input_stage_refused;

    // ---- Profiling ------------------------------------------------------
    //
    // Everything below is dead weight unless TP_PROFILING is set: `m_profiling`
    // gates every read and folds to a literal in a build without debug caps.

    struct Stages {
        NS set_inputs;
        NS infer;
        NS collect;
    };

    void accumulate_dispatch_spread();
    void report_dispatch_spread();
    void report_breakdown(const Stages& stages);

    bool m_profiling = false;

    std::vector<NS> m_rank_time;
    std::vector<NS> m_rank_start;

    NS m_dispatch_spread;
    NS m_dispatch_first;

    DumpPeriod m_dump_period;
};

}  // namespace tp_gpu
}  // namespace ov
