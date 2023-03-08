// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "itt.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"
#include "template/config.hpp"
#include "transformations/utils/utils.hpp"

// ! [executable_network:ctor_cnnnetwork]
ov::template_plugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                  const std::shared_ptr<const ov::IPlugin>& plugin,
                                                  const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                                                  const Configuration& cfg)
    : ov::ICompiledModel(model, plugin, task_executor),  // Disable default threads creation
      _cfg(cfg),
      m_model(model) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        compile_model(m_model);
    } catch (const InferenceEngine::Exception& e) {
        // Some transformations can throw legacy exception
        throw ov::Exception(e.what());
    } catch (const std::exception& e) {
        OPENVINO_ASSERT(false, "Standard exception from compilation library: ", e.what());
    } catch (...) {
        throw ov::Exception("Generic exception is thrown");
    }
}
// ! [executable_network:ctor_cnnnetwork]

// ! [executable_network:map_graph]
// forward declaration
void transform_model(const std::shared_ptr<ov::Model>& model);

void ov::template_plugin::CompiledModel::compile_model(const std::shared_ptr<ov::Model>& model) {
    // apply plugins transformations
    transform_model(model);
    // Perform any other steps like allocation and filling backend specific memory handles and so on
}
// ! [executable_network:map_graph]

// ! [executable_network:create_infer_request]
std::shared_ptr<ov::IAsyncInferRequest> ov::template_plugin::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(
        std::static_pointer_cast<ov::template_plugin::InferRequest>(internal_request),
        get_task_executor(),
        get_template_plugin()->_waitExecutor,
        get_callback_executor());

    return async_infer_request;
}

std::shared_ptr<ov::ISyncInferRequest> ov::template_plugin::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const ov::template_plugin::CompiledModel>(shared_from_this()));
}
// ! [executable_network:create_infer_request]

void ov::template_plugin::CompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}
ov::RemoteContext ov::template_plugin::CompiledModel::get_context() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::template_plugin::CompiledModel::get_runtime_model() const {
    return m_model;
}

std::shared_ptr<const ov::template_plugin::Plugin> ov::template_plugin::CompiledModel::get_template_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto template_plugin = std::static_pointer_cast<const ov::template_plugin::Plugin>(plugin);
    OPENVINO_ASSERT(template_plugin);
    return template_plugin;
}

// ! [executable_network:get_config]
ov::Any ov::template_plugin::CompiledModel::get_property(const std::string& name) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::optimal_number_of_infer_requests};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::id,
                                                    ov::enable_profiling,
                                                    ov::template_plugin::throughput_streams};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    // TODO: return more supported values for metrics
    if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();
        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        return to_string_vector(metrics);
    } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        auto configs = default_rw_properties();
        auto streamExecutorConfigKeys = ov::threading::IStreamsExecutor::Config{}
                                            .get_property(ov::supported_properties.name())
                                            .as<std::vector<std::string>>();
        for (auto&& configKey : streamExecutorConfigKeys) {
            configs.emplace_back(configKey);
        }
        return to_string_vector(configs);
    } else if (ov::model_name == name) {
        auto model_name = m_model->get_friendly_name();
        return decltype(ov::model_name)::value_type(model_name);
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = _cfg._streamsExecutorConfig._streams;
        return decltype(ov::optimal_number_of_infer_requests)::value_type(value);
    }

    if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    }

    return _cfg.Get(name);
}
// ! [executable_network:get_config]

// ! [executable_network:export]
void ov::template_plugin::CompiledModel::export_model(std::ostream& modelStream) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "ExecutableNetwork::Export");

    std::stringstream xmlFile, binFile;
    ov::pass::Serialize serializer(xmlFile, binFile);
    serializer.run_on_model(m_model);

    auto m_constants = binFile.str();
    auto m_model = xmlFile.str();

    auto dataSize = static_cast<std::uint64_t>(m_model.size());
    modelStream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    modelStream.write(m_model.c_str(), dataSize);

    dataSize = static_cast<std::uint64_t>(m_constants.size());
    modelStream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    modelStream.write(reinterpret_cast<char*>(&m_constants[0]), dataSize);
}
// ! [executable_network:export]
