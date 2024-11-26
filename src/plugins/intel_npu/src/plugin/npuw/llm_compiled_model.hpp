// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "compiled_model.hpp"

namespace ov {
namespace npuw {

class LLMInferRequest;
class LLMCompiledModel : public ov::npuw::ICompiledModel {
    using GetPropertiesMap =
        std::map<std::string, std::tuple<ov::PropertyMutability, std::function<ov::Any(const ::intel_npu::Config&)>>>;

public:
    struct KVCacheDesc {
        uint32_t max_prompt_size = 0u;
        uint32_t total_size = 0u;
        uint32_t num_stored_tokens = 0u;
        uint32_t dim = 0u;
    };

    LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const ov::AnyMap& properties);
    LLMCompiledModel() = delete;
    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    friend class LLMInferRequest;

    std::shared_ptr<ov::ISyncInferRequest> create_llm_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    void implement_properties();

    std::shared_ptr<::intel_npu::OptionsDesc> m_options_desc;
    ::intel_npu::Config m_cfg;
    GetPropertiesMap m_prop_to_opt;

    KVCacheDesc m_kvcache_desc;
    std::shared_ptr<ov::npuw::CompiledModel> m_kvcache_compiled;
    std::shared_ptr<ov::npuw::CompiledModel> m_prefill_compiled;
};

}  // namespace npuw
}  // namespace ov
