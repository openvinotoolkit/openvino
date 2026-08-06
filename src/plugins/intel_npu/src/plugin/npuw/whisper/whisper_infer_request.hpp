// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../llm_infer_request.hpp"

namespace ov {
namespace npuw {

class WhisperInferRequest final : public LLMInferRequest {
public:
    struct whisper_layer_names {
        static constexpr const char* qk_scores = "cross_attention_qk_scaled_scores";
        static constexpr const char* qk_scores_ = "cross_attention_qk_scaled_scores_";
    };

    explicit WhisperInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model)
        : LLMInferRequest(compiled_model) {}

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

protected:
    void prepare_for_new_conversation() override;

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids, ov::SoPtr<ov::ITensor> enc_hidden_states);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids);

    bool m_need_copy_kvcache = false;
    std::map<std::string, ov::SoPtr<ov::ITensor>> m_alignment_tensors{};
};

}  // namespace npuw
}  // namespace ov
