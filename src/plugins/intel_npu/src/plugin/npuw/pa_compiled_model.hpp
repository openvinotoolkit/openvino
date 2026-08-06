// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "npuw/compiled_model.hpp"

namespace ov::npuw {

class PACompiledModel;

// Dispatches the dynamic, stateless PagedAttention model deployed by the
// GenAI continuous-batching pipeline. Each dispatch is validated against the
// PA control-tensor contract (past_lens /
// subsequence_begins / block_indices(_begins) / max_context_len /
// sampled_tokens_indices), then forwarded 1:1 to a single inner infer request.
// Chunked execution over semi-static variants arrives with a later change.
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
    // One dispatch's control tensors, parsed and validated.
    struct Dispatch {
        std::vector<int64_t> past_lens;
        std::vector<int64_t> subsequence_begins;
        std::vector<int64_t> block_indices;
        std::vector<int64_t> block_indices_begins;
        std::vector<int64_t> sampled_tokens_indices;
        int64_t n_tokens = 0;
        int64_t n_seqs = 0;
    };

    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;
    // Validates the control tensors of one dispatch and parses them out.
    Dispatch validate_dispatch_locked();

    std::shared_ptr<const PACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    ov::SoPtr<ov::IAsyncInferRequest> m_inner_request;

    // Inner input ports by tensor name, for reading the control tensors.
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_inner_inputs;
    // Outer port (keyed by its node) -> matching inner port, both directions.
    std::unordered_map<const ov::Node*, ov::Output<const ov::Node>> m_port_map;
    std::size_t m_dispatch_idx = 0u;
};

// The front-end for the dynamic, stateless PagedAttention model deployed by
// the GenAI continuous-batching pipeline. This class settles the plugin-level
// seams: the model is compiled 1:1 on NPUW_PA_DEVICE (CPU by default), the
// NPU*-prefixed properties are held at this level while everything else is
// forwarded to the executing device, and the device-resolved KV cache
// geometry is stamped back onto the exposed ports so the pipeline's
// KVCacheManager reads the real cache precision and block shape.
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
