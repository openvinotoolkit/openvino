// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "llm_compiled_model.hpp"
#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest final : public ov::ISyncInferRequest {
public:
    explicit LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model,
                             const ov::npuw::LLMCompiledModel::KVCacheDesc& kvcache_desc);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void check_tensors() const override{};

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        return {};
    }

private:
    void prepare_for_new_conversation();

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> position_ids);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                        ov::SoPtr<ov::ITensor> attention_mask,
                        ov::SoPtr<ov::ITensor> position_ids);

    std::shared_ptr<ov::IAsyncInferRequest> m_kvcache_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
    LLMCompiledModel::KVCacheDesc m_kvcache_desc;
    ov::SoPtr<ov::ITensor> m_logits;
    bool m_need_copy_kvcache = false;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_out_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_out_ports;
};

}  // namespace npuw
}  // namespace ov
