// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "npuw/compiled_model.hpp"
#include "pa_dispatch.hpp"

namespace ov::npuw {

// The front-end for the dynamic, stateless PagedAttention model deployed by
// the GenAI continuous-batching pipeline. The model is compiled 1:1 on the PA
// fallback device (CPU; the internal OPENVINO_NPUW_PA_DEVICE env var exists
// for development), NPU*-prefixed properties are held at this level while
// everything else is forwarded to the executing device.
//
// The exposed ports are the inner compiled model's own ports, so the cache
// geometry the pipeline's KVCacheManager reads off them (element types,
// block shapes) is the device-resolved truth with no copying involved.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    // The wrapper adds no I/O of its own -- it exposes the inner model's ports.
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    ov::SoPtr<ov::ICompiledModel> m_compiled_model;

    // Pre-compiled semi-static token-size variants keyed by fixed token dim
    // (1024, 128, 1); the infer request dispatches token chunks onto these.
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> m_semi_static_models;

    // KV cache block size as fixed by the device at compile time; 0 if the
    // compiled cache shape is still dynamic in that dimension. Consumed by
    // the per-dispatch block-table validation.
    std::size_t m_block_size = 0u;
};

// The dispatching request. The ports are shared with the inner request (see
// PACompiledModel), so tensors travel between the request levels without
// translation. Each dispatch is validated against the PA control-tensor
// contract (past_lens / subsequence_begins / block_indices(_begins) /
// max_context_len / sampled_tokens_indices), then executed per subsequence by
// greedily routing token chunks through the pre-compiled semi-static variants
// (largest first; the 1-token variant serves the generation case). A residual
// chunk that no static size fits, or any dispatch outside the supported input
// contract, goes through the dynamic base model unchanged.
//
// Chunks only fix the activation size -- the context stays dynamic, so the
// KV cache is always addressed through the caller's block tables and no
// padding is ever written.
class PAInferRequest final : public ov::ISyncInferRequest {
public:
    PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                   ov::SoPtr<ov::IAsyncInferRequest> inner_request,
                   std::size_t block_size,
                   const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // A chunk-capable request (semi-static variant or the dynamic tail
    // request) with its ports resolved by name once.
    struct ChunkRequest {
        ov::SoPtr<ov::IAsyncInferRequest> request;
        std::unordered_map<std::string, ov::Output<const ov::Node>> inputs;
        ov::Output<const ov::Node> logits;
    };

    // Copies one dispatch's control tensors out of the inner request.
    pa::Dispatch parse_dispatch() const;
    // Per-dispatch I/O trace (Verbose): one line per input (or output) tensor
    // with a compact data digest.
    void log_dispatch_io(bool outputs) const;

    void infer_chunked(const pa::Dispatch& d);
    // Executes `n_chunk_tokens` of subsequence `seq` starting at token
    // `seq_offset` on `chunk`, scattering any sampled logits rows into
    // m_chunked_logits.
    void run_chunk(ChunkRequest& chunk, const pa::Dispatch& d, int64_t seq, int64_t seq_offset, int64_t n_chunk_tokens);

    ov::SoPtr<ov::IAsyncInferRequest> m_inner_request;

    // Input ports by tensor name, for reading the control tensors.
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_inputs_by_name;
    std::size_t m_block_size = 0u;

    // Semi-static chunk requests keyed by token size, largest first, plus a
    // dynamic request for residual chunks. These are separate from
    // m_inner_request, which holds the caller's dispatch tensors and stays
    // untouched by chunked execution.
    std::map<std::size_t, ChunkRequest, std::greater<std::size_t>> m_chunk_requests;
    // The variants' fixed token sizes, for the pa::variants_serve routing call.
    std::vector<std::size_t> m_variant_token_dims;
    ChunkRequest m_tail_request;

    // Chunked-execution result for the current dispatch; get_tensor() serves
    // it instead of the (not inferred) inner request's logits.
    ov::SoPtr<ov::ITensor> m_chunked_logits;
    bool m_serve_chunked_logits = false;
    const ov::Node* m_logits_node = nullptr;

    // Only infer() and the get_tensor() of the caller consuming its results
    // touch the members above, under the usual one-request-one-user contract,
    // so no lock is needed.
    std::size_t m_dispatch_idx = 0u;
};

}  // namespace ov::npuw
