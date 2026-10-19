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

namespace pa {

// Throws with the reason when the model is outside the flat-token contract the
// static variants implement: the known control inputs, 1-D token streams, the
// shared block table, the sampled-token gather and one logits output.
void check_static_contract(const std::shared_ptr<ov::Model>& model);

// The variant of the PA model for chunk_size tokens and sampled_dim logits
// rows. Everything but the PA ops is static. Each PA op sits in a dynamic
// island that sees only the rows subsequence_begins accounts for:
//
//   rows = Range(0, subsequence_begins[-1])
//   attn = PagedAttention(Gather(q, rows), Gather(k, rows), Gather(v, rows), ...)
//   out  = ScatterUpdate(zeros[chunk_size, H*S], rows, attn)
//
// so the padding rows never reach the attention or the KV cache.
std::shared_ptr<ov::Model> derive_static_variant(const std::shared_ptr<ov::Model>& model,
                                                 std::size_t chunk_size,
                                                 std::size_t sampled_dim);

}  // namespace pa

// Front-end for the dynamic PA model the GenAI CB pipeline deploys. Only the
// static variants are compiled, on the PA device (CPU). The ports are the
// model's own, with the KV cache geometry the device resolved, which the
// pipeline allocates the cache from.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    // Keyed by chunk size, largest first.
    using Variants = std::map<std::size_t, ov::SoPtr<ov::ICompiledModel>, std::greater<std::size_t>>;

    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    PACompiledModel(Variants&& variants,
                    const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin);

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    Variants m_variants;
};

// Validates each dispatch and runs it as chunks over the flat token axis, one
// static variant infer per chunk. The caller's tensors live in this request.
class PAInferRequest final : public ov::ISyncInferRequest {
public:
    PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                   const PACompiledModel::Variants& variants);

    void infer() override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    // The token streams and the sampled-token gather have fixed sizes, so
    // their tensors are bound once and rewritten per chunk.
    struct ChunkRequest {
        ov::SoPtr<ov::IAsyncInferRequest> request;
        std::unordered_map<std::string, ov::Output<const ov::Node>> inputs;
        ov::Output<const ov::Node> logits;
        ov::SoPtr<ov::ITensor> input_ids, position_ids, sampled_tokens_indices;
    };

    pa::Dispatch parse_dispatch() const;
    void log_dispatch_io(bool outputs) const;

    void run_chunk(ChunkRequest& chunk, const pa::Dispatch& d, const pa::Chunk& c);

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_inputs_by_name;

    std::map<std::size_t, ChunkRequest> m_chunk_requests;
    std::vector<std::size_t> m_chunk_sizes;

    // No lock: one request, one user.
    std::size_t m_dispatch_idx = 0u;
};

}  // namespace ov::npuw
