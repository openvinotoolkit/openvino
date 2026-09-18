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

// The scoring tags (NPUW_TEXT_RERANK / NPUW_TEXT_EMBED) that request the wrap.
// Recorded on the wrapper and written into the batched blob header.
struct ScoringTags {
    bool text_rerank = false;
    bool text_embed = false;
};

// Extract the scoring tags from the compile properties.
ScoringTags scoring_tags(const ov::AnyMap& properties);

// Decorator that adds batch > 1 execution on top of a batch-1 inner compiled
// model. Like failsafe and accuracy_checked it forwards everything to the inner
// and exposes the inner's own ports. A [N, ...] inference is unrolled into N
// [1, ...] inferences with a state reset between rows, so it is only valid for
// single-shot scoring (rerank, embedding), never for generation. Applied at
// ICompiledModel::create(); the blob carries a batched header in front of the
// inner blob so import rebuilds the wrap on its own.
class CompiledModel final : public ov::npuw::ICompiledModel {
public:
    // Consumes the batched header, imports the nested inner blob and wraps it.
    static std::shared_ptr<ov::npuw::ICompiledModel> import_model(std::istream& stream,
                                                                  const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                  const ov::AnyMap& properties);

    CompiledModel(const std::shared_ptr<ov::npuw::ICompiledModel>& inner,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ScoringTags& tags);

    // The inner model's ports, the wrapper adds no I/O of its own.
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    // Scoring tags come from the wrapper, everything else from the inner.
    ov::Any get_property(const std::string& name) const override;

    void release_memory() override;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

private:
    std::shared_ptr<ov::npuw::ICompiledModel> m_inner;
    ScoringTags m_tags;
};

// Unrolls one [N, ...] inference over the batch-1 inner request. Inputs with a
// leading dim of N are sliced per row, a leading dim of 1 means broadcast. The
// inner state is reset before every row and the row outputs are copied into the
// [N, ...] public outputs, which are resized in place with set_shape() when bound.
class InferRequest final : public ov::ISyncInferRequest {
public:
    InferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                 std::shared_ptr<ov::IAsyncInferRequest> inner_request);

    void infer() override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // Public input tensors of one infer() call and the batch size they agree on.
    struct BatchedInputs {
        std::vector<ov::SoPtr<ov::ITensor>> tensors;  // parallel to get_inputs()
        std::size_t batch = 1;
    };

    // Derive N from the public inputs: every leading dim is N or 1, else throw.
    BatchedInputs extract_batch() const;

    // Size the public outputs to [batch, ...] once the first row has produced the
    // inner shapes: a fresh tensor when unset, set_shape() in place when bound.
    void prepare_outputs(std::size_t batch);

    std::shared_ptr<ov::IAsyncInferRequest> m_inner;
    mutable std::mutex m_mutex;

    using MS = ov::npuw::perf::metric<ov::npuw::perf::MSec>;
    ov::npuw::perf::Profile<MS> m_profile;
};

}  // namespace ov::npuw::batched
