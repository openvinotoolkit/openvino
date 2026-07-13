// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "npuw/compiled_model.hpp"

namespace ov::npuw {

class PACompiledModel;

// PA front-end Stage 0 (CVS-190137): a dispatcher over the dynamic, stateless
// PagedAttention model the GenAI continuous-batching pipeline deploys. Each
// dispatch is validated against the PA control-tensor contract (past_lens /
// subsequence_begins / block_indices(_begins) / max_context_len /
// sampled_tokens_indices), then executed per subsequence by greedily routing
// token chunks through the pre-compiled semi-static variants (largest first;
// the 1-token variant serves the generation case). A residual chunk that no
// static size fits, or any dispatch outside the supported input contract,
// goes through the dynamic base model unchanged.
//
// Chunks only fix the activation size -- the context stays dynamic, so the
// KV cache is always addressed through the caller's block tables and no
// padding is ever written. NPU enters later by flipping NPUW_PA_DEVICE.
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
        int64_t max_context_len = -1;
        int64_t n_tokens = 0;
        int64_t n_seqs = 0;
        bool has_sti = false;
    };

    // A chunk-capable request (semi-static variant or the dynamic tail
    // request) with its ports resolved by name once.
    struct ChunkRequest {
        ov::SoPtr<ov::IAsyncInferRequest> request;
        std::unordered_map<std::string, ov::Output<const ov::Node>> inputs;
        ov::Output<const ov::Node> logits;
    };

    const ov::Output<const ov::Node>& map_port_locked(const ov::Output<const ov::Node>& port) const;
    // Validates the control tensors of one dispatch and parses them out.
    Dispatch validate_dispatch_locked();
    void validate_output_locked(int64_t expected_logits_rows);

    // True when this dispatch fits the chunked-execution input contract.
    bool can_chunk_locked(const Dispatch& d) const;
    void infer_chunked_locked(const Dispatch& d);
    // Executes `n_chunk_tokens` of subsequence `seq` starting at token
    // `seq_offset` on `chunk`, scattering any sampled logits rows into
    // m_chunked_logits.
    void run_chunk_locked(ChunkRequest& chunk,
                          const Dispatch& d,
                          int64_t seq,
                          int64_t seq_offset,
                          int64_t n_chunk_tokens);

    std::shared_ptr<const PACompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    ov::SoPtr<ov::IAsyncInferRequest> m_inner_request;

    // Semi-static chunk requests keyed by token size, largest first, plus a
    // dynamic request for residual chunks. These are separate from
    // m_inner_request, which holds the caller's dispatch tensors and stays
    // untouched by chunked execution.
    std::map<std::size_t, ChunkRequest, std::greater<std::size_t>> m_chunk_requests;
    ChunkRequest m_tail_request;

    // Chunked-execution result for the current dispatch; get_tensor() serves
    // it instead of the (not inferred) inner request's logits.
    ov::SoPtr<ov::ITensor> m_chunked_logits;
    bool m_serve_chunked_logits = false;
    const ov::Node* m_logits_node = nullptr;

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
        std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> semi_static_compiled;
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
    // Pre-compiled semi-static token-size variants keyed by fixed token dim
    // (1024, 128, 1); the infer request dispatches token chunks onto these.
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> m_semi_static_models;

    // KV cache block size as fixed by the device at compile time; 0 if the
    // compiled cache shape is still dynamic in that dimension.
    std::size_t m_block_size = 0u;
};

}  // namespace ov::npuw
