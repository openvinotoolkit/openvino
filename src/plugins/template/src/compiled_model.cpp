// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "converter_utils.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "plugin.hpp"
#include "template/config.hpp"
#include "template_async_infer_request.hpp"
#include "transformations/utils/utils.hpp"

using namespace TemplatePlugin;

namespace {

InferenceEngine::SizeVector get_dims(const ov::Output<ov::Node>& port) {
    InferenceEngine::SizeVector dims = {};
    const auto& p_shape = port.get_partial_shape();
    if (p_shape.is_static())
        dims = p_shape.get_shape();
    return dims;
}

}  // namespace

namespace ov {
namespace legacy_convert {

void fill_input_info(const ov::Output<ov::Node>& input, InferenceEngine::InputInfo::Ptr& input_info) {
    if (!input_info) {
        // Create input info
        auto param_name = input.get_node()->get_friendly_name();
        auto dims = get_dims(input);
        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(input.get_element_type()),
                                         dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        auto data = std::make_shared<InferenceEngine::Data>(param_name, desc);
        input_info = std::make_shared<InferenceEngine::InputInfo>();
        input_info->setInputData(data);
    }
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_preproc");
    if (it != rt_info.end()) {
        input_info->getPreProcess() = it->second.as<InferenceEngine::PreProcessInfo>();
    }
    it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        auto td = it->second.as<InferenceEngine::TensorDesc>();
        input_info->getInputData()->reshape(td.getDims(), td.getLayout());
        input_info->setPrecision(td.getPrecision());
    }
}
void fill_output_info(const ov::Output<ov::Node>& output, InferenceEngine::DataPtr& output_info) {
    if (!output_info) {
        // Create input info
        const auto& res_name = ov::op::util::create_ie_output_name(output);
        auto dims = get_dims(output);
        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(output.get_element_type()),
                                         dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        output_info = std::make_shared<InferenceEngine::Data>(res_name, desc);
    }
    auto& rt_info = output.get_rt_info();
    auto it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        auto td = it->second.as<InferenceEngine::TensorDesc>();
        output_info->reshape(td.getDims(), td.getLayout());
        output_info->setPrecision(td.getPrecision());
    }
}
}  // namespace legacy_convert
}  // namespace ov

// ! [executable_network:ctor_cnnnetwork]
TemplatePlugin::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                                             const Configuration& cfg)
    : ov::ICompiledModel(model, plugin, task_executor),  // Disable default threads creation
      _cfg(cfg),
      m_model(model) {
    // TODO: if your plugin supports device ID (more that single instance of device can be on host machine)
    // you should select proper device based on KEY_DEVICE_ID or automatic behavior
    // In this case, _waitExecutor should also be created per device.
    try {
        compile_model(m_model);
    } catch (const InferenceEngine::Exception&) {
        throw;
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

void TemplatePlugin::CompiledModel::compile_model(const std::shared_ptr<ov::Model>& model) {
    // apply plugins transformations
    transform_model(model);
    // Generate backend specific blob mappings. For example Inference Engine uses not ngraph::Result nodes friendly name
    // as inference request output names but the name of the layer before.
    // TODO: Remove it
    size_t idx = 0;
    for (auto&& result : model->get_results()) {
        const auto& input = result->input_value(0);
        auto name = ov::op::util::get_ie_output_name(input);
        if (_outputIndex.emplace(name, idx).second)
            idx++;
    }
    for (auto&& parameter : model->get_parameters()) {
        _inputIndex.emplace(parameter->get_friendly_name(), model->get_parameter_index(parameter));
    }

    // Perform any other steps like allocation and filling backend specific memory handles and so on
}
// ! [executable_network:map_graph]

// ! [executable_network:create_infer_request]
std::shared_ptr<ov::IAsyncInferRequest> TemplatePlugin::CompiledModel::create_infer_request() const {
    // auto internal_request = create_sync_infer_request();
    std::vector<std::shared_ptr<const ov::Node>> _inputs, _outputs;
    for (const auto& output : m_model->inputs()) {
        _inputs.emplace_back(output.get_node_shared_ptr());
    }
    for (const auto& output : outputs()) {
        _outputs.emplace_back(output.get_node_shared_ptr());
    }

    auto internal_request = std::make_shared<TemplateInferRequest>(
        _inputs,
        _outputs,
        std::static_pointer_cast<const TemplatePlugin::CompiledModel>(shared_from_this()));
    auto async_infer_request = std::make_shared<TemplateAsyncInferRequest>(
        std::static_pointer_cast<TemplatePlugin::TemplateInferRequest>(internal_request),
        get_task_executor(),
        get_template_plugin()->_waitExecutor,
        get_callback_executor());

    async_infer_request->setPointerToExecutableNetworkInternal(
        ov::legacy_convert::convert_compiled_model(std::const_pointer_cast<ov::ICompiledModel>(shared_from_this())));

    return ov::legacy_convert::convert_infer_request(async_infer_request);
}

std::shared_ptr<ov::ISyncInferRequest> TemplatePlugin::CompiledModel::create_sync_infer_request() const {
    OPENVINO_NOT_IMPLEMENTED;
    // std::vector<std::shared_ptr<const ov::Node>> _inputs, _outputs;
    // for (const auto& output : m_model->inputs()) {
    //     _inputs.emplace_back(output.get_node_shared_ptr());
    // }
    // for (const auto& output : outputs()) {
    //     _outputs.emplace_back(output.get_node_shared_ptr());
    // }
    //
    // return std::make_shared<TemplateInferRequest>(
    //     _inputs,
    //     _outputs,
    //     std::static_pointer_cast<const TemplatePlugin::CompiledModel>(shared_from_this()));
}
// ! [executable_network:create_infer_request]

void TemplatePlugin::CompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}
ov::RemoteContext TemplatePlugin::CompiledModel::get_context() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> TemplatePlugin::CompiledModel::get_runtime_model() const {
    return m_model;
}

std::shared_ptr<const Plugin> TemplatePlugin::CompiledModel::get_template_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto template_plugin = std::static_pointer_cast<const TemplatePlugin::Plugin>(plugin);
    OPENVINO_ASSERT(template_plugin);
    return template_plugin;
}

// ! [executable_network:get_config]
InferenceEngine::Parameter TemplatePlugin::CompiledModel::get_property(const std::string& name) const {
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
        auto streamExecutorConfigKeys = InferenceEngine::IStreamsExecutor::Config{}.SupportedKeys();
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
void TemplatePlugin::CompiledModel::export_model(std::ostream& modelStream) const {
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
