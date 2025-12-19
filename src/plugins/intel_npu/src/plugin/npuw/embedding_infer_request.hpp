// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "base_sync_infer_request.hpp"
#include "llm_compiled_model.hpp"
#include "llm_eagle3_extension.hpp"
#include "llm_lora_states.hpp"
#include "llm_prefix_caching.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

class EmbeddingInferRequest : public ov::ISyncInferRequest {
public:
    struct layer_names {
        static constexpr const char* input_ids = "input_ids";
        static constexpr const char* inputs_embeds = "inputs_embeds";
        static constexpr const char* attention_mask = "attention_mask";
        static constexpr const char* position_ids = "position_ids";
        static constexpr const char* past_key_values = "past_key_values";
        static constexpr const char* output_embeds = "npuw_output_embed";
        static constexpr const char* logits = "logits";
        static constexpr const char* token_type_ids = "token_type_ids";
        static constexpr const char* gemma_sliding_mask = "npuw_gemma_sliding_mask";
    };

    struct layer_ids {
        static constexpr uint32_t INPUT_IDS_SEQ_LEN_DIM = 1;
        static constexpr std::size_t kStartOutputKVCacheLayers = 1;
    };

    explicit EmbeddingInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void check_tensors() const override {};

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

protected:
    ov::SoPtr<ov::ITensor> create_prefill_output_tensor();
    void prepare_for_new_conversation();

    void infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids, ov::SoPtr<ov::ITensor> attention_mask);

    void infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids,
                             ov::SoPtr<ov::ITensor> attention_mask,
                             ov::SoPtr<ov::ITensor> input_token_ids);

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> input_token_ids);

    void update_kvcache_for(std::shared_ptr<ov::IAsyncInferRequest> request,
                            const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                            const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
                            uint32_t tokens,
                            bool v_transposed);

protected:
    std::shared_ptr<LLMCompiledModel> m_npuw_llm_compiled_model;

private:
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_out_ports;
    std::vector<ov::Output<const ov::Node>> m_prefill_past_kv_ports;

    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;

    ov::SoPtr<ov::ITensor> m_input_ids_in_tensor;
    ov::SoPtr<ov::ITensor> m_attn_mask_in_tensor;
    ov::SoPtr<ov::ITensor> m_pos_ids_in_tensor;
    ov::SoPtr<ov::ITensor> m_type_ids_in_tensor;

    ov::SoPtr<ov::ITensor> m_prefill_output;
};

}  // namespace npuw
}  // namespace ov
