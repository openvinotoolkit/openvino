// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "npuw/compiled_model.hpp"

namespace ov::npuw {

class PACompiledModel;

// PA front-end Stage 0 (CVS-190137): a 1:1 dispatcher over the dynamic,
// stateless PagedAttention model the GenAI continuous-batching pipeline
// deploys. Every dispatch is forwarded untouched to one inner infer request;
// on the way through, the PA control-tensor contract (past_lens /
// subsequence_begins / block_indices(_begins) / max_context_len /
// sampled_tokens_indices) is validated and traced.
//
// No chunking, no derived semi-static models: the inner model is compiled
// as-is on a dynamic-shape-capable device (NPUW_PA_DEVICE, default CPU).
// NPU dynamic-shape support is provided externally and slots in by flipping
// that device.
class PAInferRequest final : public ov::ISyncInferRequest {
public:
    explicit PAInferRequest(std::shared_ptr<const PACompiledModel> compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;
    // Validates the control tensors of one dispatch; returns the expected
    // number of logits rows (-1 when the model has no sampled_tokens_indices).
    int64_t validate_dispatch_locked();
    void validate_output_locked(int64_t expected_logits_rows);

    std::shared_ptr<const PACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    ov::SoPtr<ov::IAsyncInferRequest> m_inner_request;

    // Inner input ports by tensor name, for reading the control tensors.
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_inner_inputs;
    // Outer port (keyed by its node) -> matching inner port, both directions.
    std::unordered_map<const ov::Node*, ov::Output<const ov::Node>> m_port_map;
    std::size_t m_dispatch_idx = 0u;
};

class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    // The inner model is compiled first and the resolved KV cache element
    // types / shapes are stamped back onto the model's cache Parameters, so
    // this compiled model's ports expose the real geometry. The CB pipeline's
    // KVCacheManager reads precision and block shape off these ports.
    struct PreparedState {
        std::shared_ptr<ov::Model> model;
        ov::SoPtr<ov::ICompiledModel> compiled;
        std::string device;
    };
    static PreparedState prepare(const std::shared_ptr<ov::Model>& model,
                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                 const ov::AnyMap& properties);

    PACompiledModel(PreparedState prepared, const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    friend class PAInferRequest;

    std::string m_device;
    ov::SoPtr<ov::ICompiledModel> m_compiled_model;

    // KV cache block size as fixed by the device at compile time; 0 if the
    // compiled cache shape is still dynamic in that dimension.
    std::size_t m_block_size = 0u;
};

}  // namespace ov::npuw
