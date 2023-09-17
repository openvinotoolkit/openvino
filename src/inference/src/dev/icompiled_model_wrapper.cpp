// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "icompiled_model_wrapper.hpp"

#include "dev/converter_utils.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/core/except.hpp"

InferenceEngine::ICompiledModelWrapper::ICompiledModelWrapper(
    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model)
    : ov::ICompiledModel(nullptr, ov::legacy_convert::convert_plugin(model->_plugin), nullptr, nullptr),
      m_model(model) {
    std::vector<ov::Output<const ov::Node>> inputs, outputs;
    for (const auto& input : m_model->getInputs()) {
        inputs.emplace_back(input->output(0));
    }
    for (const auto& output : m_model->getOutputs()) {
        outputs.emplace_back(output->output(0));
    }
    m_inputs = inputs;
    m_outputs = outputs;
}

std::shared_ptr<ov::IAsyncInferRequest> InferenceEngine::ICompiledModelWrapper::create_infer_request() const {
    auto infer_request = m_model->CreateInferRequest();
    infer_request->setPointerToSo(m_model->GetPointerToSo());
    return ov::legacy_convert::convert_infer_request(infer_request, m_model->_plugin->GetName())._ptr;
}

void InferenceEngine::ICompiledModelWrapper::export_model(std::ostream& model) const {
    try {
        m_model->Export(model);
    } catch (const InferenceEngine::NotImplemented& ex) {
        OPENVINO_ASSERT_HELPER(ov::NotImplemented, "", false, ex.what());
    }
}

std::shared_ptr<const ov::Model> InferenceEngine::ICompiledModelWrapper::get_runtime_model() const {
    return m_model->GetExecGraphInfo();
}

void InferenceEngine::ICompiledModelWrapper::set_property(const ov::AnyMap& properties) {
    m_model->SetConfig(properties);
}

ov::Any InferenceEngine::ICompiledModelWrapper::get_property(const std::string& name) const {
    if (ov::loaded_from_cache == name) {
        return m_model->isLoadedFromCache();
    }

    auto get_supported_properties = [&]() {
        auto ro_properties = m_model->GetMetric(METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
        auto rw_properties = m_model->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
        std::vector<ov::PropertyName> supported_properties;
        for (auto&& ro_property : ro_properties) {
            if (ro_property != METRIC_KEY(SUPPORTED_METRICS) && ro_property != METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
                supported_properties.emplace_back(ro_property, ov::PropertyMutability::RO);
            }
        }
        for (auto&& rw_property : rw_properties) {
            supported_properties.emplace_back(rw_property, ov::PropertyMutability::RW);
        }
        supported_properties.emplace_back(ov::supported_properties.name(), ov::PropertyMutability::RO);
        supported_properties.emplace_back(ov::loaded_from_cache.name(), ov::PropertyMutability::RO);
        return supported_properties;
    };

    if (ov::supported_properties == name) {
        try {
            auto supported_properties = m_model->GetMetric(name).as<std::vector<ov::PropertyName>>();
            supported_properties.erase(std::remove_if(supported_properties.begin(),
                                                      supported_properties.end(),
                                                      [](const ov::PropertyName& name) {
                                                          return name == METRIC_KEY(SUPPORTED_METRICS) ||
                                                                 name == METRIC_KEY(SUPPORTED_CONFIG_KEYS);
                                                      }),
                                       supported_properties.end());
            return supported_properties;
        } catch (ov::Exception&) {
            return get_supported_properties();
        } catch (InferenceEngine::Exception&) {
            return get_supported_properties();
        }
    }
    try {
        return m_model->GetMetric(name);
    } catch (ov::Exception&) {
        return m_model->GetConfig(name);
    } catch (InferenceEngine::Exception&) {
        return m_model->GetConfig(name);
    }
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>
InferenceEngine::ICompiledModelWrapper::get_executable_network() {
    return m_model;
}

std::shared_ptr<const InferenceEngine::IExecutableNetworkInternal>
InferenceEngine::ICompiledModelWrapper::get_executable_network() const {
    return m_model;
}
