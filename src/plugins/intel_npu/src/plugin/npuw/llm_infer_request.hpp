// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "llm_compiled_model.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

template <>
struct std::hash<ov::Output<const ov::Node>> {
    std::size_t operator()(const ov::Output<const ov::Node>& port) const {
        using std::hash;
        using std::size_t;
        using std::string;

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:

        return ((hash<std::size_t>()(port.get_index()) ^ (hash<const ov::Node*>()(port.get_node()) << 1)) >> 1) ^
               (hash<std::string>()(port.get_any_name()) << 1);
    }
};

namespace ov {
namespace npuw {

class LLMInferRequest final : public ov::ISyncInferRequest {
public:
    struct layer_names {
        static constexpr const char* input_ids = "input_ids";
        static constexpr const char* inputs_embeds = "inputs_embeds";
        static constexpr const char* attention_mask = "attention_mask";
        static constexpr const char* position_ids = "position_ids";
        static constexpr const char* past_key_values = "past_key_values";
        static constexpr const char* output_embeds = "npuw_output_embed";
        static constexpr const char* logits = "logits";
    };

    explicit LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void check_tensors() const override {};

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

private:
    void prepare_for_new_conversation();

    void clear_chunk_prefill_kv_cache();

    void init_tensor(const ov::Output<const ov::Node>& port);
    void copy_kvcache();
    void update_kvcache_for(std::shared_ptr<ov::IAsyncInferRequest> request,
                            std::unordered_map<std::string, ov::Output<const ov::Node>> in_ports,
                            std::unordered_map<std::string, ov::Output<const ov::Node>> out_ports,
                            uint32_t tokens);

    void infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids,
                               ov::SoPtr<ov::ITensor> attention_mask,
                               ov::SoPtr<ov::ITensor> position_ids);

    void infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                             ov::SoPtr<ov::ITensor> attention_mask,
                             ov::SoPtr<ov::ITensor> position_ids);

    void update_out_tensors_from(std::shared_ptr<ov::IAsyncInferRequest> request);

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> position_ids);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                        ov::SoPtr<ov::ITensor> attention_mask,
                        ov::SoPtr<ov::ITensor> position_ids);

    std::shared_ptr<ov::IAsyncInferRequest> m_kvcache_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
    // This infer request is optional, so can be null.
    std::shared_ptr<ov::IAsyncInferRequest> m_lm_head_request;
    std::shared_ptr<LLMCompiledModel> m_npuw_llm_compiled_model;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_out_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_out_ports;

    // NB: It can be either input_ids(LLM) or inputs_embeds(VLM)
    std::string m_input_ids_name;
    std::unordered_map<ov::Output<const ov::Node>, ov::SoPtr<ov::ITensor>> m_out_tensors;

    bool m_generate_initialized = false;
};

}  // namespace npuw
}  // namespace ov
