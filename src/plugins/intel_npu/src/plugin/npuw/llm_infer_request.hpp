// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/core/descriptor/output.hpp"

namespace ov {
namespace npuw {

class LLMCompiledModel;
class LLMInferRequest final : public ov::ISyncInferRequest {
public:
    explicit LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void check_tensors() const override{};

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const {}
    virtual std::vector<ov::SoPtr<ov::IVariableState>> query_state() const {}

private:
    void prepare_for_new_conversation();

    void infer_prefill(ov::SoPtr<ov::ITensor> input_ids,
                       ov::SoPtr<ov::ITensor> attention_mask,
                       ov::SoPtr<ov::ITensor> position_ids);

    void infer_generate(ov::SoPtr<ov::ITensor> input_ids,
                        ov::SoPtr<ov::ITensor> attention_mask,
                        ov::SoPtr<ov::ITensor> position_ids);

    struct KVCacheDesc {
        uint32_t max_prompt_size;
        uint32_t total_size;
        uint32_t num_stored_tokens;
        uint32_t dim;
    };

    std::shared_ptr<ov::IAsyncInferRequest> m_kvcache_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
    KVCacheDesc m_kvcache_desc;
    ov::SoPtr<ov::ITensor> m_logits;
    bool m_need_copy_kvcache = false;

    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_prefill_out_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_in_ports;
    std::unordered_map<std::string, ov::Output<const ov::Node>> m_kvcache_out_ports;
};

}  // namespace npuw
}  // namespace ov
