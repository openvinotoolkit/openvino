// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base_sync_infer_request.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_base_request.hpp"
#include "openvino/core/descriptor/output.hpp"

namespace ov {
namespace npuw {

class EmbeddingInferRequest : public ov::npuw::LLMInferBaseRequest {
public:
    explicit EmbeddingInferRequest(const std::shared_ptr<LLMCompiledModel>& compiled_model);

    void infer() override;
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

protected:
    ov::SoPtr<ov::ITensor> create_prefill_output_tensor();
    void prepare_for_new_conversation();

    void infer_chunked_prefill(ov::SoPtr<ov::ITensor> input_ids, ov::SoPtr<ov::ITensor> attention_mask);

    void infer_whole_prefill(ov::SoPtr<ov::ITensor> input_ids, ov::SoPtr<ov::ITensor> attention_mask);

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids, ov::SoPtr<ov::ITensor> attention_mask);

private:
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_out_ports;
    std::vector<ov::Output<const ov::Node>> m_prefill_past_kv_ports;

    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;

    ov::SoPtr<ov::ITensor> m_input_ids_in_tensor;
    ov::SoPtr<ov::ITensor> m_attn_mask_in_tensor;
    ov::SoPtr<ov::ITensor> m_pos_ids_in_tensor;

    ov::SoPtr<ov::ITensor> m_prefill_output;
};

}  // namespace npuw
}  // namespace ov
