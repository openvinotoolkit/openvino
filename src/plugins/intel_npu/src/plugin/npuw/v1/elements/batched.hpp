// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "../../compiled_model.hpp"
#include "../../perf.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov::npuw::batched {

// True when the compile/import properties opt into batched single-shot scoring --
// currently the text rerank and text embedding pipelines. Used by the entry points
// (npuw::ICompiledModel::create and the NPUW import path) to decide whether to
// apply the batched element.
bool requested(const ov::AnyMap& properties);

// A compiled-model decorator that adds batched (batch > 1) execution on top of an
// inner compiled model that only supports a batch size of 1 (as NPUW's LLM
// pipeline does, since it reshapes every sub-model to a static batch of 1).
//
// Like the other v1/elements wrappers (failsafe, accuracy_checked) it is a
// transparent decorator that exposes the inner model's I/O (inputs()/outputs()
// forward to the inner, so the ports are literally the same objects) and forwards
// everything but inference to the inner model. Unlike them it is a *fan-out*
// element: a single [N, ...] inference is unrolled into N independent [1, ...]
// inferences on the inner request, the inner variable state (KV-cache) is reset
// between rows, and the per-row outputs are written into rows of the [N, ...]
// public output tensors.
//
// This is correct for single-shot scoring workloads whose rows are independent
// -- text reranking and text embedding -- where batched and per-row results are
// identical and batching is purely a throughput/ergonomics choice. It is NOT
// valid for autoregressive generation, where state must persist across calls.
//
// It is instantiated at the NPUW entry points only: npuw::ICompiledModel::create()
// on compilation and the plugin's NPUW import path on blob import (the element is
// runtime-only and is not part of the serialized blob). Both wrap the
// LLMCompiledModel produced there from the very same model, which is what
// guarantees the inner is the matching batch-1 compilation.
class CompiledModel final : public ov::npuw::ICompiledModel {
public:
    // Factory method. Returns inner unwrapped when enabled == false, keeping the
    // default path zero-overhead.
    static std::shared_ptr<ov::npuw::ICompiledModel> create(const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                                                            const std::shared_ptr<const ov::IPlugin>& plugin,
                                                            bool enabled);

    CompiledModel(const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                  const std::shared_ptr<const ov::IPlugin>& plugin);

    // The wrapper adds no I/O of its own -- it exposes the inner model's ports.
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    void release_memory() override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    std::shared_ptr<ov::npuw::ICompiledModel> m_inner;
};

// Sync infer request that unrolls a batched inference over the single-sequence
// inner request.
//
// The public input tensors default to the inner request's own tensors (surfaced in
// the constructor), so a plain batch-1 infer works exactly as on the inner. When
// the caller binds [N, ...] inputs, infer() takes N as the largest leading
// dimension across the inputs (an input with a leading dim of 1 is shared across
// rows), and for each row: resets the inner variable state, binds the row's
// [1, ...] view of every batched input, runs the inner request, and copies the
// inner outputs into row i of the [N, ...] public output tensors. Caller-bound
// output tensors are reused when they already have the right shape and type.
class InferRequest final : public ov::ISyncInferRequest {
public:
    InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                 std::shared_ptr<ov::IAsyncInferRequest> inner_request);

    void infer() override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // The public input tensors snapshotted for one infer() call, together with the
    // batch size derived from them.
    struct BatchedInputs {
        std::vector<ov::SoPtr<ov::ITensor>> tensors;  // parallel to get_inputs()
        std::size_t batch = 1;
    };

    // Snapshot the public inputs and derive the batch size: the largest leading
    // dimension across them. Every input must either carry the batch ([N, ...],
    // sliced per row by infer()) or be shared across rows ([1, ...], bound whole);
    // anything else throws.
    BatchedInputs extract_batch() const;

    // Make the public output tensors [batch, ...] copies of the inner's [1, ...]
    // outputs, reusing caller-bound tensors that already fit. The wrapped model's
    // ports are dynamic, so this can only run once the first row has been scored
    // and the inner output shapes are known.
    void ensure_batched_outputs(std::size_t batch);

    std::shared_ptr<ov::IAsyncInferRequest> m_inner;
    mutable std::mutex m_mutex;

    // Per-phase timings of the unroll.
    using MS = ov::npuw::perf::metric<ov::npuw::perf::MSec>;
    ov::npuw::perf::Profile<MS> m_profile;
};

}  // namespace ov::npuw::batched
