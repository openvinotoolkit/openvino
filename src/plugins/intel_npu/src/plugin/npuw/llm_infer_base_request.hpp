// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_compiled_model.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

class LLMInferBaseRequest : public ov::ISyncInferRequest {
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

    explicit LLMInferBaseRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model)
        : ISyncInferRequest(compiled_model),
          m_npuw_llm_compiled_model(compiled_model) {}

    void check_tensors() const override {};
    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

protected:
    void update_kvcache_for(std::shared_ptr<ov::IAsyncInferRequest> request,
                            const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                            const std::unordered_map<std::string, ov::Output<const ov::Node>>& out_ports,
                            uint32_t num_tokens,
                            bool v_transposed);
    void init_tensor(const ov::Output<const ov::Node>& port);
    void init_ports();

protected:
    std::shared_ptr<LLMCompiledModel> m_npuw_llm_compiled_model;
};

}  // namespace npuw
}  // namespace ov
