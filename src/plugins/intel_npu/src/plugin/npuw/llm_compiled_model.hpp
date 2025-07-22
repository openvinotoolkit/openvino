// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "compiled_model.hpp"

namespace ov {
namespace npuw {

class VariableState final : public ov::IVariableState {
public:
    explicit VariableState(const std::string& name, const ov::SoPtr<ov::ITensor>& tensor) : ov::IVariableState(name) {
        m_state = tensor;
    }

    virtual void set_state(const ov::SoPtr<ov::ITensor>& newState) override {
#if 0
        if (newState->get_byte_size() != m_state->get_byte_size()) {
            OPENVINO_THROW("Byte size mismatch");
        }

        std::memcpy(m_state->data(), newState->data(), newState->get_byte_size());
#endif
        m_state = newState;
    }

    virtual void reset() override {
        std::memset(m_state->data(), 0, m_state->get_byte_size());
    }

    ~VariableState() override = default;
};

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
        bool v_tensors_transposed = false;
    };

    LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const ov::AnyMap& properties);
    LLMCompiledModel(const std::shared_ptr<ov::Model>& model,
                     const std::shared_ptr<const ov::IPlugin>& plugin,
                     const bool serialized);
    LLMCompiledModel() = delete;

    void export_model(std::ostream& model) const override;
    static std::shared_ptr<LLMCompiledModel> import_model(std::istream& stream,
                                                          const std::shared_ptr<const ov::IPlugin>& plugin,
                                                          const ov::AnyMap& properties);

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    friend class LLMInferRequest;

    std::shared_ptr<ov::ISyncInferRequest> create_llm_infer_request();
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    void implement_properties();

    void serialize(std::ostream& stream, const ov::npuw::s11n::CompiledContext& ctx) const;
    static std::shared_ptr<LLMCompiledModel> deserialize(std::istream& stream,
                                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                                         const ov::AnyMap& properties,
                                                         const ov::npuw::s11n::CompiledContext& ctx);

    std::string m_name;
    std::shared_ptr<::intel_npu::OptionsDesc> m_options_desc;
    ::intel_npu::Config m_cfg;
    GetPropertiesMap m_prop_to_opt;
    ov::AnyMap m_non_llm_props;

    // Cache bf16 constants for weightless deserialization
    ov::npuw::s11n::BF16Cache m_bf16_consts;

    KVCacheDesc m_kvcache_desc;
    std::shared_ptr<ov::npuw::CompiledModel> m_kvcache_compiled;
    std::shared_ptr<ov::npuw::CompiledModel> m_prefill_compiled;

    uint64_t m_prefill_chunk_size;

    void convertStatefulLoRAtoStateless(std::shared_ptr<ov::Model>& model);
    mutable std::vector<ov::SoPtr<ov::IVariableState>> m_variableStates;
};

}  // namespace npuw
}  // namespace ov
