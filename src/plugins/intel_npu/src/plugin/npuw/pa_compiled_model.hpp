// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "npuw/compiled_model.hpp"
#include "pa_dispatch.hpp"

namespace ov::npuw {

// The front-end for the dynamic, stateless PagedAttention model deployed by
// the GenAI continuous-batching pipeline. The dynamic model itself is never
// compiled: every dispatch runs on static variants derived from it, each with
// a fixed token count and a fixed number of logits rows. Inside a variant the
// PagedAttention ops sit in a small dynamic island (Range, Gather,
// ScatterUpdate driven by subsequence_begins), so the padding rows of a
// variant never reach the attention and its KV cache.
//
// The exposed ports are the dynamic model's own, with the KV cache geometry
// (element types, block shapes) stamped from the compiled variants: that is
// what the pipeline's KVCacheManager reads to allocate the cache pools. The
// variants run on the PA fallback device (CPU; the internal
// OPENVINO_NPUW_PA_DEVICE env var exists for development). NPU*-prefixed
// properties are held at this level, everything else is forwarded.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    // The flat-token LLM contract the static variants implement: the known
    // control inputs only, 1-D token streams, the shared block table, the
    // sampled-token gather and one logits output with static row geometry.
    // Throws with the reason otherwise. Static so it is unit-testable.
    static void require_static_contract(const std::shared_ptr<ov::Model>& model);

    // A static variant of the PA model for token_dim tokens and sampled_dim
    // logits rows. The token streams and the sampled-token gather get fixed
    // sizes, so the whole transformer is static, and every PagedAttention op
    // is wrapped in a dynamic island that only sees the rows
    // subsequence_begins accounts for:
    //
    //   n     = subsequence_begins[-1]
    //   rows  = Range(0, n)
    //   attn  = PagedAttention(Gather(q, rows), Gather(k, rows), Gather(v, rows), ...)
    //   out   = ScatterUpdate(zeros[token_dim, H*Sv], rows, attn)
    //
    // The padding rows of a variant thus never reach the attention or its
    // KV cache, and come out of the island as zeros. Everything outside the
    // islands is static; on the CPU device the islands run with their real
    // shapes. Static so it is unit-testable.
    static std::shared_ptr<ov::Model> derive_static_variant(const std::shared_ptr<ov::Model>& base_model,
                                                            std::size_t token_dim,
                                                            std::size_t sampled_dim);

private:
    struct Prepared {
        std::shared_ptr<ov::Model> model;
        std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> variants;
        std::size_t block_size = 0u;
    };
    static Prepared prepare(const std::shared_ptr<ov::Model>& model,
                            const std::shared_ptr<const ov::IPlugin>& plugin,
                            const ov::AnyMap& properties);
    PACompiledModel(Prepared&& prepared, const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    // The dynamic model with the cache geometry stamped; owns the exposed ports.
    std::shared_ptr<ov::Model> m_model;

    // Static variants keyed by token count.
    std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>> m_variants;

    // The KV cache block size the block tables are validated against.
    std::size_t m_block_size = 0u;
};

// The dispatching request. The caller's tensors live in this request; each
// dispatch is validated against the PA control-tensor contract, planned into
// variant infers (pa::plan_dispatch) and executed on the variant requests,
// which share the caller's KV cache pools and see the caller's controls
// rebased per chunk. The sampled logits rows are assembled back into one
// output in the caller's order.
class PAInferRequest final : public ov::ISyncInferRequest {
public:
    PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                   std::size_t block_size,
                   const std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>>& variants);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // A variant request with its ports resolved by name once. The fixed-size
    // inputs (the token streams and the sampled-token gather) are allocated
    // once and rewritten per chunk; the KV cache pools are bound whenever
    // the caller sets them.
    struct VariantRequest {
        ov::SoPtr<ov::IAsyncInferRequest> request;
        std::unordered_map<std::string, ov::Output<const ov::Node>> inputs;
        ov::Output<const ov::Node> logits;
        ov::SoPtr<ov::ITensor> input_ids, position_ids, sampled_tokens_indices;
        std::size_t token_dim = 0u;
        std::size_t sampled_dim = 0u;
    };

    // Copies one dispatch's control tensors out of the caller's tensors.
    pa::Dispatch parse_dispatch() const;
    // Per-dispatch I/O trace (Verbose): one line per input (or output) tensor
    // with a compact data digest.
    void log_dispatch_io(bool outputs) const;
    // Runs one chunk on its variant, scattering the sampled logits rows into
    // m_logits.
    void run_chunk(VariantRequest& variant, const pa::Dispatch& d, const pa::Chunk& chunk);

    // Input ports by tensor name, for reading the caller's tensors.
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_inputs_by_name;
    std::size_t m_block_size = 0u;

    std::map<std::size_t, VariantRequest> m_variants;
    std::vector<std::size_t> m_variant_token_dims;
    // The most logits rows one variant produces.
    std::size_t m_max_sampled = 0u;

    // The current dispatch's logits, served by get_tensor() for the logits port.
    ov::SoPtr<ov::ITensor> m_logits;
    const ov::Node* m_logits_node = nullptr;

    // Only infer() and the get_tensor() of the caller consuming its results
    // touch the members above, under the usual one-request-one-user contract,
    // so no lock is needed.
    std::size_t m_dispatch_idx = 0u;
};

}  // namespace ov::npuw
