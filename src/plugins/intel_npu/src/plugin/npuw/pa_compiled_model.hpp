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

// Front-end for the dynamic PagedAttention model the GenAI CB pipeline
// deploys. The model is compiled 1:1 on the PA device (CPU) next to its
// semi-static variants. The ports are the inner model's, so the pipeline reads
// the device-resolved KV cache geometry off them.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

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
