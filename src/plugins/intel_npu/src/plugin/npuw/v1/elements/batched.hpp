// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov::npuw::batched {

class InferRequest;

// A compiled-model wrapper that adds batched (batch > 1) execution on top of an
// inner compiled model that only supports a batch size of 1 (as NPUW's LLM
// pipeline does, since it reshapes every sub-model to a static batch of 1).
//
// Like the other v1/elements wrappers (failsafe, accuracy_checked) it is a
// transparent ov::ICompiledModel decorator that exposes the same I/O as the
// model it wraps and forwards property queries to the inner model.  Unlike
// them it is a *fan-out* element: a single [N, ...] inference is unrolled into
// N independent [1, ...] inferences on the inner request, the inner variable
// state (KV-cache) is reset between rows, and the per-row outputs are stacked
// back into [N, ...] tensors along the batch dimension (axis 0).
//
// This is correct for single-shot scoring workloads whose rows are independent
// -- text reranking and text embedding -- where batched and per-row results are
// identical and batching is purely a throughput/ergonomics choice.  It is NOT
// valid for autoregressive generation, where rows are not independent.
//
// It composes on top of the other elements, e.g.:
//
//   Batched( AccuracyChecked( Failsafe(NPU -> CPU) ) )
//
// and the inner request needs no awareness of batching at all -- a stock
// single-sequence LLM/embedding infer request works unchanged.
class CompiledModel final : public ov::ICompiledModel {
public:
    // Factory method.  Returns inner_compiled unwrapped when enabled == false,
    // keeping the zero-overhead path trivial (mirrors accuracy_checked::create).
    static ov::SoPtr<ov::ICompiledModel> create(const std::shared_ptr<ov::Model>& model,
                                                const std::shared_ptr<const ov::IPlugin>& plugin,
                                                ov::SoPtr<ov::ICompiledModel> inner_compiled,
                                                bool enabled);

    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  ov::SoPtr<ov::ICompiledModel> inner_compiled);

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

private:
    friend class InferRequest;

    ov::SoPtr<ov::ICompiledModel> m_inner;
};

// Sync infer request that unrolls a batched inference over a single-sequence
// inner request.
//
// On each infer() call it determines the batch size N from the inputs, then for
// each row 0..N-1: resets the inner request's variable state, binds the row's
// [1, ...] slice of every input, runs the inner request, and copies the inner
// [1, ...] output into row i of the wrapper's [N, ...] output tensors.  The
// public input/output tensors are held by the ISyncInferRequest base; only the
// inner request sees [1, ...] shapes.
//
// It is constructed from the compiled model whose I/O it exposes plus the inner
// request to drive.  This lets it be produced both by Batched::CompiledModel
// (standalone, composable element) and reused directly by a pipeline that
// already owns a single-sequence request (e.g. NPUW's LLMCompiledModel wrapping
// its LLMInferRequest for scoring), without that pipeline having to wrap its
// whole compiled model.
class InferRequest final : public ov::ISyncInferRequest {
public:
    // Drive an async inner request (used by the standalone Batched::CompiledModel decorator,
    // where the inner is a separate compiled model with its own task executor).
    InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                 std::shared_ptr<ov::IAsyncInferRequest> inner_request);

    // Drive a sync inner request directly, on the calling thread, without an executor. Used when a
    // pipeline wraps its own single-sequence request (e.g. NPUW's LLMCompiledModel wrapping its
    // LLMInferRequest): this avoids a nested-executor deadlock that would otherwise occur if the
    // inner ran on the same task executor as the outer (async) request.
    InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                 std::shared_ptr<ov::ISyncInferRequest> inner_request);

    void infer() override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    void init_public_tensors();

    // The inner single-sequence request, driven row by row. Exactly one of the two handles is set;
    // the small accessors below hide which interface (sync/async) is in use from infer().
    const std::vector<ov::Output<const ov::Node>>& inner_inputs() const;
    const std::vector<ov::Output<const ov::Node>>& inner_outputs() const;
    void inner_set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor);
    ov::SoPtr<ov::ITensor> inner_get_tensor(const ov::Output<const ov::Node>& port) const;
    void inner_infer();
    std::vector<ov::SoPtr<ov::IVariableState>> inner_query_state() const;

    std::shared_ptr<ov::IAsyncInferRequest> m_inner_async;
    std::shared_ptr<ov::ISyncInferRequest> m_inner_sync;
    mutable std::mutex m_mutex;
};

}  // namespace ov::npuw::batched
