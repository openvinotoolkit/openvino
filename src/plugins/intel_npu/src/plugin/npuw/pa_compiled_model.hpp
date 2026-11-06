// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "llm_compiled_model_utils.hpp"
#include "npuw/compiled_model.hpp"
#include "pa_dispatch.hpp"

namespace ov::npuw {

namespace pa {

// The PA model is token-major: the flat token count sits in dimension 0, where
// the NPU compiler reads a batch and lowers the trunk into batched layers.
// Every token-major tensor also carries a singleton dimension ([tokens, 1,
// hidden] through the FFN, [tokens, heads, 1, head_size] around RoPE), so
// consuming it and prepending a batch of 1 moves the token count out of the
// batch position without moving data. This does that around the weight
// MatMuls and the elementwise arithmetic of a variant with `tokens` > 1, so
// the NPU gets batch-1 layers. The reshapes are metadata only.
void relayout_token_major(const std::shared_ptr<ov::Model>& model, std::size_t tokens);

// A model with tied embeddings decompresses one weight table for both the
// embedding Gather and the LM head MatMul. Partitioned as is, the head
// subgraph outputs the whole decompressed table every inference and the tail
// takes it as an input, so the LM head weights are no longer constants. This
// gives every MatMul whose constant-only weight decompression it shares with
// other consumers its own copy of the decompression ops.
void untie_matmul_weights(const std::shared_ptr<ov::Model>& model);

}  // namespace pa

// Front-end for the dynamic PagedAttention model the GenAI CB pipeline
// deploys. The model is compiled 1:1 on the PA device (CPU) next to its
// semi-static variants, which NPUW partitions: the repeated blocks fold onto
// NPU and every PA op stays on the PA device. The ports are the inner model's,
// so the pipeline reads the device-resolved KV cache geometry off them.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    // The NPUW config the variants compile with: the NPU defaults of the LLM
    // path plus the partitioning above, under the caller's properties, with one
    // weights bank for all.
    static ov::AnyMap make_variant_config(const ov::AnyMap& properties,
                                          const std::string& pa_device,
                                          const std::string& weights_bank,
                                          const std::optional<util::NPUDesc>& npudesc);

    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    ov::SoPtr<ov::ICompiledModel> m_compiled_model;

    // Keyed by chunk size.
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> m_semi_static_models;
};

// Validates each dispatch, then runs it 1:1 or splits every subsequence into
// chunks over the variants, largest first, with a dynamic tail. Chunks fix only
// the token count: the cache is addressed through the caller's block tables,
// so nothing is padded.
class PAInferRequest final : public ov::ISyncInferRequest {
public:
    PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                   ov::SoPtr<ov::IAsyncInferRequest> inner_request,
                   const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    struct ChunkRequest {
        ov::SoPtr<ov::IAsyncInferRequest> request;
        std::unordered_map<std::string, ov::Output<const ov::Node>> inputs;
        ov::Output<const ov::Node> logits;
    };

    pa::Dispatch parse_dispatch() const;
    void log_dispatch_io(bool outputs) const;

    void infer_chunked(const pa::Dispatch& d);
    void run_chunk(ChunkRequest& chunk, const pa::Dispatch& d, int64_t seq, int64_t seq_offset, int64_t n_chunk_tokens);

    ov::SoPtr<ov::IAsyncInferRequest> m_inner_request;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_inputs_by_name;

    // Largest first. The caller's tensors stay in m_inner_request.
    std::map<std::size_t, ChunkRequest, std::greater<std::size_t>> m_chunk_requests;
    std::vector<std::size_t> m_chunk_sizes;
    ChunkRequest m_tail_request;

    // The chunked path's result, served by get_tensor().
    ov::SoPtr<ov::ITensor> m_chunked_logits;
    bool m_serve_chunked_logits = false;
    const ov::Node* m_logits_node = nullptr;

    // No lock: one request, one user.
    std::size_t m_dispatch_idx = 0u;
};

}  // namespace ov::npuw
