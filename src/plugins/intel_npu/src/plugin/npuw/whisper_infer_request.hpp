// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_infer_request.hpp"

namespace ov {
namespace npuw {

class WhisperInferRequest final : public LLMInferRequest {
public:
    explicit WhisperInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model) 
        : LLMInferRequest(compiled_model) {}

    void infer() override;

protected:
    void prepare_for_new_conversation() override;

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> enc_hidden_states);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids);

    bool m_need_copy_kvcache = false;
};

}  // namespace npuw
}  // namespace ov
