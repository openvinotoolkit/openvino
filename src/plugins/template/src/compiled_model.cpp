// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "plugin.hpp"
#include "template/config.hpp"
#include "template_itt.hpp"
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
                                             const ov::AnyMap& cfg)
    : ov::ICompiledModel(model, plugin, task_executor),  // Disable default threads creation
      m_model(model) {
    // Init properties
    get_properties()
        .add(m_rw_properties.m_properties)
        .add(ov::common_property(ov::model_name),
             [this]() {
                 return decltype(ov::model_name)::value_type(m_model->get_friendly_name());
             })
        .add(ov::common_property(ov::optimal_number_of_infer_requests),
             std::ref(m_rw_properties._streamsExecutorConfig._streams));
    get_properties()
        .set(cfg)
        // IF we need to make some properties read only use ov::PropertyAccess::ro() with property name
        .ro(ov::streams::num)
        .ro(ov::stream_executor_property);
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
    // Perform any other steps like allocation and filling backend specific memory handles and so on
}
// ! [executable_network:map_graph]

// ! [executable_network:create_infer_request]
std::shared_ptr<ov::IAsyncInferRequest> TemplatePlugin::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request =
        std::make_shared<AsyncInferRequest>(std::static_pointer_cast<TemplatePlugin::InferRequest>(internal_request),
                                            get_task_executor(),
                                            get_template_plugin()->_waitExecutor,
                                            get_callback_executor());

    return async_infer_request;
}

std::shared_ptr<ov::ISyncInferRequest> TemplatePlugin::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const TemplatePlugin::CompiledModel>(shared_from_this()));
}
// ! [executable_network:create_infer_request]

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
